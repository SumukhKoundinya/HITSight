import cv2
import numpy as np
import mediapipe as mp


class HeadTracker:
    """
    Lightweight head-pose tracker using MediaPipe Face Mesh.

    Output per frame:
        yaw, pitch, roll
        tx, ty, tz (relative face translation, approximate)
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Approximate 3D facial reference points.
        # These are generic reference coordinates; they are used
        # consistently to estimate relative head pose.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # nose
            (0.0, -330.0, -65.0),     # chin
            (-225.0, 170.0, -135.0),  # left eye
            (225.0, 170.0, -135.0),   # right eye
            (-150.0, -150.0, -125.0), # left mouth
            (150.0, -150.0, -125.0)   # right mouth
        ], dtype=np.float64)

        # MediaPipe landmark indices corresponding approximately
        # to the six points above.
        self.landmark_indices = [
            1,    # nose
            152,  # chin
            33,   # left eye
            263,  # right eye
            61,   # left mouth
            291   # right mouth
        ]

    def process(self, frame):
        """
        Process one BGR OpenCV frame.

        Returns:
            dict or None
        """

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        image_points = np.array([
            (
                landmarks[i].x * w,
                landmarks[i].y * h
            )
            for i in self.landmark_indices
        ], dtype=np.float64)

        # Approximate camera parameters.
        focal_length = w
        center = (w / 2.0, h / 2.0)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        # Convert rotation vector → rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extract Euler angles.
        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2 +
            rotation_matrix[1, 0] ** 2
        )

        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2]
            )

            yaw = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            roll = np.arctan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0]
            )
        else:
            pitch = np.arctan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1]
            )

            yaw = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            roll = 0

        return {
            "yaw": float(np.degrees(yaw)),
            "pitch": float(np.degrees(pitch)),
            "roll": float(np.degrees(roll)),
            "tx": float(translation_vector[0]),
            "ty": float(translation_vector[1]),
            "tz": float(translation_vector[2]),
        }

    def draw(self, frame, pose):
        """
        Draw simple head-pose information on the frame.
        """

        if pose is None:
            cv2.putText(
                frame,
                "Face not detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            return frame

        text = (
            f"Yaw: {pose['yaw']:.1f}  "
            f"Pitch: {pose['pitch']:.1f}  "
            f"Roll: {pose['roll']:.1f}"
        )

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        return frame

    def close(self):
        self.face_mesh.close()
