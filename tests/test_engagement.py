import unittest
from core.face_tracker import StudentTrack
from core.engagement_analyzer import EngagementAnalyzer
from core.multimodal_fusion import MultimodalFusionEngine
import config


class TestEngagementDynamics(unittest.TestCase):
    """Unit tests for head pose, movement, audio, and multimodal fusion."""

    def setUp(self):
        self.analyzer = EngagementAnalyzer()
        self.fusion = MultimodalFusionEngine()

    def test_attentive_forward_pose(self):
        """Student looking forward with open eyes must yield high engagement."""
        track = StudentTrack("STU-001", "Focused Student")
        track.pitch = 0.0
        track.yaw = 2.0
        track.roll = 0.0
        track.eye_openness = 0.95
        track.motion_energy = 5.0

        score = self.analyzer.analyze_student_engagement(track)
        self.assertGreaterEqual(score, 85.0)
        self.assertTrue(track.is_attentive)

    def test_distracted_yaw_penalty(self):
        """Looking sharply sideways (e.g. 45 degrees) must penalize score."""
        track = StudentTrack("STU-002", "Distracted Student")
        track.pitch = 0.0
        track.yaw = 45.0  # Turned away
        track.eye_openness = 0.95

        score = self.analyzer.analyze_student_engagement(track)
        self.assertLess(score, 75.0)
        self.assertFalse(track.is_attentive)
        self.assertGreater(track.distraction_count, 0)

    def test_drowsy_head_slouch_penalty(self):
        """Downward pitch (slouch/phone) and closed eyes must heavily penalize score."""
        track = StudentTrack("STU-003", "Drowsy Student")
        track.pitch = -35.0  # Downward looking
        track.eye_openness = 0.12  # Drowsy eye closure

        score = self.analyzer.analyze_student_engagement(track)
        self.assertLess(score, 50.0)
        self.assertGreater(track.drowsy_count, 0)

    def test_multimodal_fusion_calculation(self):
        """Verify CEI calculation and state mapping."""
        vis_data = {"average_visual_engagement": 80.0, "attentive_count": 4, "distracted_count": 1}
        audio_data = {"interaction_score": 70.0, "rms": 0.04, "interaction_category": "Lecture"}
        
        tracks = [StudentTrack(f"STU-{i}") for i in range(5)]
        for t in tracks:
            t.motion_energy = 8.0

        res = self.fusion.fuse(vis_data, audio_data, tracks)
        self.assertIn("cei", res)
        self.assertGreaterEqual(res["cei"], 60.0)
        self.assertIn(res["engagement_state"], ["Highly Engaged", "Moderately Engaged"])


if __name__ == "__main__":
    unittest.main()
