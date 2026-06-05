import cv2
import numpy as np
import dlib
import os

def load_known_faces(face_net, shape_predictor, face_recognition_model):
    """Load known faces for face recognition"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    known_faces_directory = os.path.join(project_root, 'known_faces')

    print("\n========== LOADING KNOWN FACES ==========")
    print(f"Known faces directory: {known_faces_directory}")

    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(known_faces_directory):
        print("No known_faces directory found")
        return known_face_encodings, known_face_names

    files = os.listdir(known_faces_directory)

    print(f"Files found: {files}")

    for file_name in files:

        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):

            try:
                print(f"\nProcessing image: {file_name}")

                image_path = os.path.join(
                    known_faces_directory,
                    file_name
                )

                image = cv2.imread(image_path)

                if image is None:
                    print(f"Failed to read image: {file_name}")
                    continue

                print(f"Image loaded successfully: {image.shape}")

                rgb_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                h, w = image.shape[:2]

                blob = cv2.dnn.blobFromImage(
                    image,
                    1.0,
                    (300, 300),
                    [104, 117, 123],
                    False,
                    False
                )

                face_net.setInput(blob)

                detections = face_net.forward()

                print(
                    f"Faces detected: {detections.shape[2]}"
                )

                face_found = False

                for i in range(detections.shape[2]):

                    confidence = detections[0, 0, i, 2]

                    print(f"Confidence: {confidence}")

                    if confidence > 0.5:

                        face_found = True

                        box = (
                            detections[0, 0, i, 3:7]
                            * np.array([w, h, w, h])
                        )

                        x1, y1, x2, y2 = box.astype(int)

                        rect = dlib.rectangle(
                            x1,
                            y1,
                            x2,
                            y2
                        )

                        shape = shape_predictor(
                            rgb_image,
                            rect
                        )

                        encoding = np.array(
                            face_recognition_model.compute_face_descriptor(
                                rgb_image,
                                shape
                            )
                        )

                        known_face_encodings.append(
                            encoding
                        )

                        person_name = os.path.splitext(
                            file_name
                        )[0]

                        known_face_names.append(
                            person_name
                        )

                        print(
                            f"Successfully loaded face: {person_name}"
                        )

                        break

                if not face_found:
                    print(
                        f"No valid face found in {file_name}"
                    )

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print("\n========== LOADING COMPLETE ==========")
    print(f"Total known faces loaded: {len(known_face_names)}")

    return known_face_encodings, known_face_names

def recognize_faces(
    frame,
    face_net,
    shape_predictor,
    face_recognition_model,
    known_face_encodings,
    known_face_names
):
    """Process face recognition on a frame"""

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        1.0,
        (300, 300),
        [104, 117, 123],
        False,
        False
    )

    face_net.setInput(blob)

    detections = face_net.forward()

    result_frame = frame.copy()

    current_names = set()

    unknown_count = 0

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = (
                detections[0, 0, i, 3:7]
                * np.array([w, h, w, h])
            )

            x1, y1, x2, y2 = box.astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 > 0 and y2 - y1 > 0:

                rect = dlib.rectangle(
                    x1,
                    y1,
                    x2,
                    y2
                )

                shape = shape_predictor(
                    rgb_frame,
                    rect
                )

                face_encoding = np.array(
                    face_recognition_model.compute_face_descriptor(
                        rgb_frame,
                        shape
                    )
                )

                name = "Unknown"

                if len(known_face_encodings) > 0:

                    distances = np.linalg.norm(
                        known_face_encodings - face_encoding,
                        axis=1
                    )

                    min_distance = np.min(distances)

                    if min_distance < 0.55:

                        index = np.argmin(distances)

                        name = known_face_names[index]

                        current_names.add(name)

                    else:
                        unknown_count += 1

                color = (
                    (0, 255, 0)
                    if name != "Unknown"
                    else (0, 0, 255)
                )

                cv2.rectangle(
                    result_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    result_frame,
                    name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    if unknown_count > 0:

        current_names.add(
            f"{unknown_count} unknown face{'s' if unknown_count > 1 else ''}"
        )

    return result_frame, current_names