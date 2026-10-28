import cv2
import time
import numpy as np
from typing import Optional, Generator, Tuple, List
import config
from core.face_tracker import FaceTracker, StudentTrack
from core.engagement_analyzer import EngagementAnalyzer


class VideoProcessor:
    """
    Video Ingestion, Preprocessing, and Visual Annotation Engine.
    Handles webcam streams, video files, and browser frame ingestion.
    """

    def __init__(self, face_tracker: FaceTracker, engagement_analyzer: EngagementAnalyzer):
        self.face_tracker = face_tracker
        self.engagement_analyzer = engagement_analyzer
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_active = False
        self.current_source = "camera"
        self.anonymize_faces = config.ANONYMIZE_FACES_DEFAULT
        self.frame_count = 0
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_tracks: List[StudentTrack] = []

    def start_camera(self, device_index: int = config.DEFAULT_CAMERA_INDEX) -> bool:
        """Initialize camera capture."""
        if self.cap is not None:
            self.stop_camera()

        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            print(f"[VideoProcessor] Warning: Camera at index {device_index} could not be opened.")
            self.is_camera_active = False
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_FRAME_HEIGHT)
        self.is_camera_active = True
        self.current_source = "camera"
        return True

    def open_video_file(self, file_path: str) -> bool:
        """Open a recorded classroom video file."""
        if self.cap is not None:
            self.stop_camera()

        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print(f"[VideoProcessor] Error: Failed to open video file {file_path}")
            return False

        self.is_camera_active = True
        self.current_source = "file"
        self.face_tracker.reset_session()
        return True

    def stop_camera(self):
        """Release camera/video capture."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_camera_active = False

    def process_single_frame(self, frame: np.ndarray, current_time: Optional[float] = None) -> Tuple[np.ndarray, List[StudentTrack]]:
        """Process one video frame: detection, tracking, engagement analysis, and annotation."""
        self.frame_count += 1
        
        # Downsample if image is excessively large for high throughput
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280.0 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))

        # Run Face Detection & Spatial Tracking
        tracks = self.face_tracker.detect_and_track(frame, current_time)
        self.latest_tracks = tracks

        # Analyze engagement for each active student
        for track in tracks:
            self.engagement_analyzer.analyze_student_engagement(track)

        # Annotate visual frame with bounding boxes, HUD badges, and pose indicators
        annotated = self._annotate_frame(frame, tracks)
        self.latest_annotated_frame = annotated

        return annotated, tracks

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Generator yielding MJPEG frame stream for Flask streaming endpoint."""
        while self.is_camera_active and self.cap is not None:
            success, frame = self.cap.read()
            if not success:
                # If reading video file, loop or terminate
                if self.current_source == "file":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    time.sleep(0.05)
                    continue

            annotated_frame, _ = self.process_single_frame(frame)

            # Encode to JPEG
            ret, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(1.0 / config.TARGET_FPS)

    def _annotate_frame(self, frame: np.ndarray, tracks: List[StudentTrack]) -> np.ndarray:
        """Render informative AR overlays on top of the video feed."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw Global HUD Header Bar
        cv2.rectangle(annotated, (0, 0), (w, 40), (20, 24, 33), -1)
        header_text = f"STUDENTS DETECTED: {len(tracks)} | PRIVACY-PRESERVING OFFLINE MODE"
        cv2.putText(annotated, header_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 2)

        for track in tracks:
            x, y, bw, bh = track.bbox
            score = track.current_engagement_score
            status = track.attendance_status

            # Privacy Option: Blur face region if requested
            if self.anonymize_faces and bw > 10 and bh > 10:
                face_roi = annotated[max(0, y):min(h, y+bh), max(0, x):min(w, x+bw)]
                if face_roi.size > 0:
                    blurred_roi = cv2.GaussianBlur(face_roi, (51, 51), 30)
                    annotated[max(0, y):min(h, y+bh), max(0, x):min(w, x+bw)] = blurred_roi

            # Color coding: Green (High Engaged), Yellow (Moderate), Red (Low)
            if score >= 70.0:
                box_color = (46, 204, 113)  # Emerald Green
            elif score >= 45.0:
                box_color = (241, 196, 15)  # Amber Yellow
            else:
                box_color = (231, 76, 60)   # Coral Red

            # Draw sleek bounding box
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), box_color, 2)

            # Draw Corner Accents
            line_len = min(15, bw // 4, bh // 4)
            cv2.line(annotated, (x, y), (x + line_len, y), box_color, 4)
            cv2.line(annotated, (x, y), (x, y + line_len), box_color, 4)
            cv2.line(annotated, (x + bw, y), (x + bw - line_len, y), box_color, 4)
            cv2.line(annotated, (x + bw, y), (x + bw, y + line_len), box_color, 4)

            # Student Name & Attendance Status Label Tag
            label_top = f"{track.name} [{status}]"
            tag_w = max(140, len(label_top) * 9)
            cv2.rectangle(annotated, (x, max(0, y - 26)), (x + tag_w, max(26, y)), box_color, -1)
            cv2.putText(
                annotated, label_top, (x + 5, max(18, y - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2
            )

            # Engagement Score & Head Pose HUD below box
            eng_label = f"Eng: {score}% | Yaw: {int(track.yaw)} P: {int(track.pitch)}"
            cv2.rectangle(annotated, (x, min(h - 5, y + bh)), (x + 195, min(h, y + bh + 22)), (30, 30, 30), -1)
            cv2.putText(
                annotated, eng_label, (x + 5, min(h - 5, y + bh + 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
            )

        return annotated
