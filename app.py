import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort
from ultralytics import YOLO

YOLO_MODEL = "yolov12l-face.pt"
ADAFACE_MODEL = "adaface-onnx/weights/adaface_ir_101.onnx"

st.set_page_config(
    page_title="Automatic Attendance System",
    page_icon="",
    layout="wide"
)

st.title("Automatic Attendance System")
st.caption("Face detection, recognition, and attendance tracking")

with st.sidebar:
    st.header("Settings")

    match_threshold = st.slider(
        "Match threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.01
    )

    bbox_expansion = st.slider(
        "Bounding-box expansion",
        min_value=0.00,
        max_value=0.20,
        value=0.08,
        step=0.01,
        format="%.2f"
    )

    st.caption(
        "Expansion is applied on each side of the YOLO box. "
        "Set it to 0 to disable expansion."
    )

    DATABASE_OPTIONS = {
    "No Expansion": "no_expansions/face_database.npz",
    "6.25% Expansion": "6.25%_expanded/face_database_6.25%_expanded.npz",
    "8% Expansion": "8%_expanded/face_database_8%_expanded.npz",
    "12.5% Expansion": "12.5%_expanded/face_database_12.5%_expanded.npz"
    }

    selected_database = st.selectbox(
        label="Select Database",
        options=list(DATABASE_OPTIONS.keys())
    )

    DATABASE_FILE = DATABASE_OPTIONS[selected_database]

@st.cache_resource
def load_models():
    if not os.path.isfile(YOLO_MODEL):
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL}"
        )

    if not os.path.isfile(ADAFACE_MODEL):
        raise FileNotFoundError(
            f"AdaFace model not found: {ADAFACE_MODEL}"
        )

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )

    yolo = YOLO(YOLO_MODEL)

    session = ort.InferenceSession(
        ADAFACE_MODEL,
        providers=providers
    )

    input_name = session.get_inputs()[0].name

    return yolo, session, input_name, providers


@st.cache_data
def load_database():
    if not os.path.isfile(DATABASE_FILE):
        raise FileNotFoundError(
            f"Face database not found: {DATABASE_FILE}"
        )

    with np.load(DATABASE_FILE, allow_pickle=False) as data:
        names = data["names"].astype(str)
        embeddings = np.asarray(
            data["embeddings"], dtype=np.float32
        )

    if (
        len(names) == 0
        or len(names) != len(embeddings)
        or embeddings.ndim != 2
    ):
        raise RuntimeError("Face database is empty or invalid.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise RuntimeError(
            "Face database contains zero-length embeddings."
        )

    embeddings = embeddings / norms

    return names.tolist(), embeddings

def get_embedding(face_image, session, input_name):
    if face_image is None or face_image.size == 0:
        return None

    image = cv2.resize(face_image, (112, 112))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = ((image - 127.5) / 128.0).transpose(2, 0, 1)[None, ...]

    embedding = np.asarray(
        session.run(None, {input_name: image})[0],
        dtype=np.float32
    ).reshape(-1)

    norm = np.linalg.norm(embedding)

    return embedding / norm if norm else None

def expand_bbox(box, image_shape, expansion):
    x1, y1, x2, y2 = map(int, box)
    height, width = image_shape[:2]

    expand_x = int((x2 - x1) * expansion)
    expand_y = int((y2 - y1) * expansion)

    return (
        max(0, x1 - expand_x),
        max(0, y1 - expand_y),
        min(width, x2 + expand_x),
        min(height, y2 + expand_y)
    )

def match_face(embedding, names, database_embeddings, threshold):
    if embedding is None:
        return "Unknown", 0.0

    similarities = database_embeddings @ embedding
    best_index = int(np.argmax(similarities))
    similarity = float(similarities[best_index])

    if similarity >= threshold:
        return names[best_index], similarity

    return "Unknown", similarity

def draw_result(image, bbox, name, similarity):
    x1, y1, x2, y2 = bbox
    distance = 1.0 - similarity

    label = f"{name} | PRESENT" if name != "Unknown" else "Unknown"
    details = f"Sim: {similarity:.3f}  Dist: {distance:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    (w1, h1), _ = cv2.getTextSize(label, font, 0.65, 2)
    (w2, h2), _ = cv2.getTextSize(details, font, 0.55, 2)

    label_width = max(w1, w2) + 12
    label_height = h1 + h2 + 25
    top = max(0, y1 - label_height)

    cv2.rectangle(
        image, (x1, y1), (x2, y2), (0, 255, 0), 2
    )

    cv2.rectangle(
        image, (x1, top), (x1 + label_width, y1),
        (0, 255, 0), -1
    )

    cv2.putText(
        image, label, (x1 + 5, max(h1, y1 - h2 - 10)),
        font, 0.65, (0, 0, 0), 2, cv2.LINE_AA
    )

    cv2.putText(
        image, details, (x1 + 5, max(h2, y1 - 5)),
        font, 0.55, (0, 0, 0), 2, cv2.LINE_AA
    )

def process_image(
    image,
    yolo,
    session,
    input_name,
    names,
    database_embeddings,
    threshold,
    expansion
):
    frame = image.copy()
    present_people = set()
    detections = []

    results = yolo(frame, verbose=False)

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            original_box = box.xyxy[0].cpu().numpy()

            bbox = expand_bbox(
                original_box, frame.shape, expansion
            )

            x1, y1, x2, y2 = bbox

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]

            embedding = get_embedding(
                face_crop, session, input_name
            )

            name, similarity = match_face(
                embedding,
                names,
                database_embeddings,
                threshold
            )

            distance = 1.0 - similarity

            if name != "Unknown":
                present_people.add(name)

            draw_result(frame, bbox, name, similarity)

            detections.append({
                "Name": name,
                "Similarity": round(similarity, 4),
                "Distance": round(distance, 4),
                "Status": (
                    "PRESENT" if name != "Unknown" else "Unknown"
                )
            })

    attendance = [
        {
            "Name": name,
            "Status": (
                "PRESENT" if name in present_people else "ABSENT"
            )
        }
        for name in sorted(names)
    ]

    return frame, attendance, detections

try:
    with st.spinner("Loading YOLO, AdaFace, and face database..."):
        yolo, session, input_name, providers = load_models()
        names, database_embeddings = load_database()

    st.sidebar.success("Models and database loaded")

except Exception as error:
    st.error(f"Could not initialize the application: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an attendance image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is None:
    st.info("Upload an image to detect faces and generate attendance.")
    st.stop()

file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if image is None:
    st.error("The uploaded image could not be read.")
    st.stop()

st.subheader("Input image")
st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

if st.button("Run Attendance", type="primary"):
    try:
        with st.spinner("Detecting and recognizing faces..."):
            result_image, attendance, detections = process_image(
                image,
                yolo,
                session,
                input_name,
                names,
                database_embeddings,
                match_threshold,
                bbox_expansion
            )

        st.session_state["result_image"] = result_image
        st.session_state["attendance"] = attendance
        st.session_state["detections"] = detections

    except Exception as error:
        st.error(f"Processing failed: {error}")


if "result_image" in st.session_state:
    result_image = st.session_state["result_image"]
    attendance = st.session_state["attendance"]

    present_count = sum(
        person["Status"] == "PRESENT"
        for person in attendance
    )
    absent_count = len(attendance) - present_count

    st.divider()
    st.header("Attendance Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Registered People", len(attendance))
    col2.metric("Present", present_count)
    col3.metric("Absent", absent_count)

    left, right = st.columns(2)

    with left:
        st.subheader("Recognized Image")
        st.image(
            cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    with right:
        st.subheader("Attendance")

        if attendance:
            st.dataframe(
                pd.DataFrame(attendance),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No registered people in the database.")

    success, encoded_image = cv2.imencode(
        ".jpg", result_image
    )

    if success:
        st.download_button(
            "Download Annotated Image",
            data=encoded_image.tobytes(),
            file_name="attendance_result.jpg",
            mime="image/jpeg"
        )

    csv_data = pd.DataFrame(attendance).to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Attendance CSV",
        data=csv_data,
        file_name="attendance.csv",
        mime="text/csv"
    )