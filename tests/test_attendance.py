import unittest
import time
from core.face_tracker import StudentTrack, FaceTracker
from core.attendance_manager import AttendanceManager
import config


class TestSustainedAttendance(unittest.TestCase):
    """Unit tests for sustained presence logic & automated attendance logs."""

    def setUp(self):
        self.mgr = AttendanceManager()

    def test_transient_passerby_rejection(self):
        """Transient appearances (e.g. 1-2 seconds) must NOT be marked Present."""
        track = StudentTrack("STU-999", "Passerby", bbox=(10, 10, 50, 50))
        track.sustained_seconds = 1.5  # Only seen for 1.5s
        
        status = track.evaluate_attendance(session_elapsed_sec=60.0)
        self.assertIn("Passerby", status)
        self.assertNotEqual(status, "Present")

    def test_sustained_presence_present(self):
        """Student present for >= 60% of session time must be marked Present."""
        track = StudentTrack("STU-101", "Diligent Student", bbox=(20, 20, 80, 80))
        track.sustained_seconds = 50.0  # 50s out of 60s (83.3%)
        
        status = track.evaluate_attendance(session_elapsed_sec=60.0)
        self.assertEqual(status, "Present")
        self.assertGreaterEqual(track.sustained_presence_ratio, 0.60)

    def test_partial_attendance(self):
        """Student present for 35% of session time must be marked Partial."""
        track = StudentTrack("STU-102", "Late Student", bbox=(20, 20, 80, 80))
        track.sustained_seconds = 21.0  # 21s out of 60s (35%)
        
        status = track.evaluate_attendance(session_elapsed_sec=60.0)
        self.assertEqual(status, "Partial Attendance")

    def test_absent_threshold(self):
        """Student present for only 10% must be marked Absent."""
        track = StudentTrack("STU-103", "Brief Student", bbox=(20, 20, 80, 80))
        track.sustained_seconds = 6.0  # 6s out of 60s (10%)
        
        status = track.evaluate_attendance(session_elapsed_sec=60.0)
        self.assertIn("Absent", status)

    def test_attendance_report_generation(self):
        """Verify report summary, counts, and output fields."""
        tracks = {
            "STU-001": StudentTrack("STU-001", "Aarav Sharma", bbox=(10, 10, 50, 50)),
            "STU-002": StudentTrack("STU-002", "Priya Patel", bbox=(60, 60, 50, 50))
        }
        tracks["STU-001"].sustained_seconds = 55.0
        tracks["STU-002"].sustained_seconds = 25.0

        report = self.mgr.generate_attendance_report(tracks, session_elapsed_sec=60.0, session_id="test_sess_01")
        self.assertEqual(report["present_count"], 1)
        self.assertEqual(report["partial_count"], 1)
        self.assertGreater(report["total_students"], 0)
        self.assertIn("records", report)


if __name__ == "__main__":
    unittest.main()
