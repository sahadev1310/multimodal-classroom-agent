"""
Core intelligence engines for Multimodal Classroom Attendance & Engagement Agent.
"""
from core.face_tracker import FaceTracker, StudentTrack
from core.engagement_analyzer import EngagementAnalyzer
from core.audio_analyzer import AudioAnalyzer
from core.multimodal_fusion import MultimodalFusionEngine
from core.teaching_agent import TeachingAgent
from core.attendance_manager import AttendanceManager
from core.video_processor import VideoProcessor

__all__ = [
    "FaceTracker",
    "StudentTrack",
    "EngagementAnalyzer",
    "AudioAnalyzer",
    "MultimodalFusionEngine",
    "TeachingAgent",
    "AttendanceManager",
    "VideoProcessor"
]
