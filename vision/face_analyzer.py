import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass


@dataclass
class FaceMetrics:
    eye_contact_percent: float
    smile_rate_percent: float
    frames_analyzed: int
    frames_with_face: int


def _has_eye_contact(landmarks, frame_width: int, frame_height: int) -> bool:
    try:
        left_iris = landmarks[468]
        left_outer = landmarks[33]
        left_inner = landmarks[133]
        right_iris = landmarks[473]
        right_outer = landmarks[263]
        right_inner = landmarks[362]

        left_eye_width = abs(left_outer.x - left_inner.x)
        left_iris_offset = abs(left_iris.x - (left_outer.x + left_inner.x) / 2)
        left_ratio = left_iris_offset / (left_eye_width + 1e-6)

        right_eye_width = abs(right_outer.x - right_inner.x)
        right_iris_offset = abs(right_iris.x - (right_outer.x + right_inner.x) / 2)
        right_ratio = right_iris_offset / (right_eye_width + 1e-6)

        return left_ratio < 0.3 and right_ratio < 0.3
    except IndexError:
        return False


def _has_smile(landmarks) -> bool:
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]

    mouth_width = abs(right_corner.x - left_corner.x)
    mouth_height = abs(top_lip.y - bottom_lip.y)
    ratio = mouth_width / (mouth_height + 1e-6)
    return ratio > 3.5


def analyze_frames(frames: list[np.ndarray]) -> FaceMetrics:
    mp_face_mesh = mp.solutions.face_mesh
    eye_contact_count = 0
    smile_count = 0
    frames_with_face = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb_frame)

            if not result.multi_face_landmarks:
                continue

            frames_with_face += 1
            landmarks = result.multi_face_landmarks[0].landmark

            if _has_eye_contact(landmarks, frame.shape[1], frame.shape[0]):
                eye_contact_count += 1
            if _has_smile(landmarks):
                smile_count += 1

    base = frames_with_face if frames_with_face > 0 else 1
    return FaceMetrics(
        eye_contact_percent=round(eye_contact_count / base * 100, 1),
        smile_rate_percent=round(smile_count / base * 100, 1),
        frames_analyzed=len(frames),
        frames_with_face=frames_with_face,
    )
