import cv2
import numpy as np


def extract_frames(video_path: str, sample_rate: int = 5) -> list[np.ndarray]:
    """
    Trích xuất frames từ video.
    sample_rate=5 nghĩa là lấy 1 frame mỗi 5 frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            frames.append(frame)
        frame_count += 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    duration_seconds = frame_count / fps
    print(f"Video: {frame_count} frames, ~{duration_seconds:.1f}s, lấy {len(frames)} frames để phân tích")
    return frames
