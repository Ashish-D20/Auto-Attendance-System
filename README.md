# Auto-Class-Attendance
A system that detects faces in an input image and matches it against the faces in database and detects whether a face/person from the database is present in the image or not.

In this repository, names like "no_expansion", "6.25%_expansion", etc. refer to the expansion percentage applied to the bounding boxes of detected faces. This is done as sometimes bounding boxes sometimes clip out some important features that might be used in matching images. Usage of expansion clearly helps as inferred from classification reports [[here]](./accuracy%20reports/Performance%20Comparison.png)

In this system, we use "yolov12l-face.pt", a Yolo V12 model fine tuned for detecting face and "adaface_ir_101.onnx" a face recognition model used for matching low quality images, for converting images into image embeddings. 
The flow goes like this:
1. Input image with a face.
2. Yolo detects faces in the images.
3. adaface converts detected faces into image embeddings.
4. input embeddings are checked against database embeddings for similarity.
5. input face embedding is labelled with embedding that matches the most in the database.

#How to run the system
1. Make a 
