import os
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# Configuration
DATABASE_FOLDER = "database"
YOLO_MODEL = "yolov12l-face.pt"
ADAFACE_MODEL = "adaface-onnx/weights/adaface_ir_101.onnx"
OUTPUT_DATABASE = "ingestion methods/face_database.npz"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Load models
if not os.path.isfile(ADAFACE_MODEL):
    raise FileNotFoundError(f"AdaFace model not found: {ADAFACE_MODEL}")

yolo = YOLO(YOLO_MODEL)

providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers()
    else ["CPUExecutionProvider"]
)

session = ort.InferenceSession(ADAFACE_MODEL, providers=providers)
input_name = session.get_inputs()[0].name


def get_embedding(face):
    if face is None or face.size == 0:
        return None

    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
    face = ((face - 127.5) / 128.0).transpose(2, 0, 1)[None, ...]

    embedding = np.asarray(
        session.run(None, {input_name: face})[0],
        dtype=np.float32
    ).reshape(-1)

    norm = np.linalg.norm(embedding)
    return embedding / norm if norm else None


def extract_largest_face(image):
    largest_crop, largest_area = None, 0

    for result in yolo(image, verbose=False):
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)

            if area > largest_area:
                largest_area = area
                largest_crop = image[y1:y2, x1:x2]

    return largest_crop


def main():
    if not os.path.isdir(DATABASE_FOLDER):
        raise FileNotFoundError(f"Database folder not found: {DATABASE_FOLDER}")

    names, embeddings = [], []

    for filename in sorted(os.listdir(DATABASE_FOLDER)):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image = cv2.imread(os.path.join(DATABASE_FOLDER, filename))

        if image is None:
            print(f"[SKIP] Cannot read: {filename}")
            continue

        face = extract_largest_face(image)

        if face is None:
            print(f"[SKIP] No face detected: {filename}")
            continue

        embedding = get_embedding(face)

        if embedding is None:
            print(f"[SKIP] Embedding failed: {filename}")
            continue

        names.append(os.path.splitext(filename)[0])
        embeddings.append(embedding)
        print(f"[OK] Processed: {filename}")

    if not embeddings:
        raise RuntimeError("No valid face embeddings were generated.")

    np.savez_compressed(
        OUTPUT_DATABASE,
        names=np.asarray(names),
        embeddings=np.stack(embeddings)
    )

    print(f"\nSaved {len(names)} face embeddings to {OUTPUT_DATABASE}")


if __name__ == "__main__":
    main()
