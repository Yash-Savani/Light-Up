from ultralytics import YOLO
from services.face_service import load_known_faces
from services.scene_service import SceneDescriber

import cv2
import dlib
import os


class ModelManager:

    def __init__(self):

        self.object_model = None
        self.face_net = None
        self.shape_predictor = None
        self.face_recognition_model = None

        self.known_face_encodings = []
        self.known_face_names = []

        self.scene_describer = None

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(
            os.path.dirname(current_dir)
        )

        self.models_dir = os.path.join(
            project_dir,
            "models"
        )

    def load_object_model(self):

        if self.object_model is None:

            print("Loading YOLO model...")

            self.object_model = YOLO("yolov8n.pt")

        return self.object_model

    def load_face_models(self):

        if self.face_net is None:

            print("Loading face detection models...")

            self.face_net = cv2.dnn.readNet(
                os.path.join(
                    self.models_dir,
                    'deploy.prototxt'
                ),
                os.path.join(
                    self.models_dir,
                    'res10_300x300_ssd_iter_140000.caffemodel'
                )
            )

            self.shape_predictor = dlib.shape_predictor(
                os.path.join(
                    self.models_dir,
                    'shape_predictor_68_face_landmarks.dat'
                )
            )

            self.face_recognition_model = dlib.face_recognition_model_v1(
                os.path.join(
                    self.models_dir,
                    'dlib_face_recognition_resnet_model_v1.dat'
                )
            )

        return (
            self.face_net,
            self.shape_predictor,
            self.face_recognition_model
        )

    def initialize_all(self):

        print("Initializing all models...")

        # Load object detector
        self.load_object_model()

        # Load face models
        (
            self.face_net,
            self.shape_predictor,
            self.face_recognition_model
        ) = self.load_face_models()

        # Load known faces
        (
            self.known_face_encodings,
            self.known_face_names
        ) = load_known_faces(
            self.face_net,
            self.shape_predictor,
            self.face_recognition_model
        )

        print(
            f"Loaded {len(self.known_face_names)} known faces"
        )

        # Load scene describer
        self.scene_describer = SceneDescriber()

        print("All models initialized successfully!")