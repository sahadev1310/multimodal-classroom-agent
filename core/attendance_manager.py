import os
import csv
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import config
from core.face_tracker import StudentTrack


class AttendanceManager:
    """
    Manages student enrollment roster, sustained presence validation,
    and automatic generation of timestamped attendance logs and reports.
    """

    def __init__(self, roster_path: Optional[str] = None):
        self.roster_file = Path(roster_path or (config.ROSTER_DIR / "roster.json"))
        self.roster: Dict[str, dict] = self._load_roster()

    def _load_roster(self) -> Dict[str, dict]:
        """Load enrolled student roster from local storage."""
        if self.roster_file.exists():
            try:
                with open(self.roster_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[AttendanceManager] Error loading roster: {e}")
        return self._create_default_roster()

    def _create_default_roster(self) -> Dict[str, dict]:
        """Create sample initial roster if none exists."""
        default_roster = {
            "STU-001": {"student_id": "STU-001", "name": "Aarav Sharma", "department": "Computer Science"},
            "STU-002": {"student_id": "STU-002", "name": "Priya Patel", "department": "Data Science"},
            "STU-003": {"student_id": "STU-003", "name": "Rohan Iyer", "department": "AI & Robotics"},
            "STU-004": {"student_id": "STU-004", "name": "Ananya Sen", "department": "Computer Science"},
            "STU-005": {"student_id": "STU-005", "name": "Vikram Rao", "department": "Electrical Eng"}
        }
        try:
            with open(self.roster_file, "w", encoding="utf-8") as f:
                json.dump(default_roster, f, indent=2)
        except Exception:
            pass
        return default_roster

    def save_roster(self):
        """Persist current roster to JSON."""
        try:
            with open(self.roster_file, "w", encoding="utf-8") as f:
                json.dump(self.roster, f, indent=2)
        except Exception as e:
            print(f"[AttendanceManager] Error saving roster: {e}")

    def enroll_student(self, student_id: str, name: str, department: str = "General") -> dict:
        """Enroll or update a student in the roster."""
        self.roster[student_id] = {
            "student_id": student_id,
            "name": name,
            "department": department,
            "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_roster()
        return self.roster[student_id]

    def remove_student(self, student_id: str) -> bool:
        """Delete student from roster."""
        if student_id in self.roster:
            del self.roster[student_id]
            self.save_roster()
            return True
        return False

    def generate_attendance_report(
        self,
        tracks: Dict[str, StudentTrack],
        session_elapsed_sec: float,
        session_id: Optional[str] = None
    ) -> dict:
        """
        Evaluate full classroom attendance with sustained presence logic.
        Merges enrolled students with detected tracks (marking non-detected enrolled students as Absent).
        """
        session_id = session_id or f"session_{int(time.time())}"
        session_date = time.strftime("%Y-%m-%d %H:%M:%S")

        records = []
        present_count = 0
        partial_count = 0
        absent_count = 0

        # Process detected tracks
        detected_ids = set()
        for track_id, track in tracks.items():
            status = track.evaluate_attendance(session_elapsed_sec)
            detected_ids.add(track_id)

            if status == "Present":
                present_count += 1
            elif status == "Partial Attendance":
                partial_count += 1
            else:
                absent_count += 1

            avg_eng = (
                float(np.mean(track.engagement_history))
                if track.engagement_history else track.current_engagement_score
            )

            record = {
                "student_id": track.track_id,
                "name": track.name,
                "status": status,
                "first_seen": time.strftime("%H:%M:%S", time.localtime(track.first_seen)),
                "last_seen": time.strftime("%H:%M:%S", time.localtime(track.last_seen)),
                "sustained_duration_sec": round(track.sustained_seconds, 1),
                "sustained_ratio_pct": round(track.sustained_presence_ratio * 100, 1),
                "avg_engagement_pct": round(avg_eng, 1),
                "distraction_count": track.distraction_count,
                "drowsy_count": track.drowsy_count
            }
            records.append(record)

        # Include enrolled students who were never detected
        for enrolled_id, profile in self.roster.items():
            if enrolled_id not in detected_ids:
                absent_count += 1
                records.append({
                    "student_id": enrolled_id,
                    "name": profile.get("name", "Unknown"),
                    "status": "Absent (Not Detected)",
                    "first_seen": "N/A",
                    "last_seen": "N/A",
                    "sustained_duration_sec": 0.0,
                    "sustained_ratio_pct": 0.0,
                    "avg_engagement_pct": 0.0,
                    "distraction_count": 0,
                    "drowsy_count": 0
                })

        total_students = len(records)
        attendance_rate = round((present_count / total_students * 100), 1) if total_students > 0 else 0.0

        summary = {
            "session_id": session_id,
            "session_date": session_date,
            "session_duration_sec": round(session_elapsed_sec, 1),
            "session_duration_formatted": time.strftime("%H:%M:%S", time.gmtime(session_elapsed_sec)),
            "total_students": total_students,
            "present_count": present_count,
            "partial_count": partial_count,
            "absent_count": absent_count,
            "attendance_rate_pct": attendance_rate,
            "records": records
        }

        # Save to local privacy-preserving storage
        self._export_to_files(summary, session_id)
        return summary

    def _export_to_files(self, summary: dict, session_id: str):
        """Export session attendance report to CSV and JSON."""
        # Save JSON
        json_path = config.ATTENDANCE_LOGS_DIR / f"{session_id}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"[AttendanceManager] JSON export error: {e}")

        # Save CSV
        csv_path = config.ATTENDANCE_LOGS_DIR / f"{session_id}.csv"
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Student ID", "Name", "Attendance Status", "First Seen",
                    "Last Seen", "Sustained Duration (s)", "Sustained Ratio (%)",
                    "Avg Engagement (%)", "Distractions", "Drowsy Events"
                ])
                for r in summary["records"]:
                    writer.writerow([
                        r["student_id"], r["name"], r["status"], r["first_seen"],
                        r["last_seen"], r["sustained_duration_sec"], r["sustained_ratio_pct"],
                        r["avg_engagement_pct"], r["distraction_count"], r["drowsy_count"]
                    ])
        except Exception as e:
            print(f"[AttendanceManager] CSV export error: {e}")

    def list_saved_sessions(self) -> List[dict]:
        """List all historical saved session reports."""
        sessions = []
        for file in config.ATTENDANCE_LOGS_DIR.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data.get("session_id", file.stem),
                        "session_date": data.get("session_date", "Unknown"),
                        "duration": data.get("session_duration_formatted", "00:00:00"),
                        "total_students": data.get("total_students", 0),
                        "present_count": data.get("present_count", 0),
                        "attendance_rate": data.get("attendance_rate_pct", 0.0),
                        "file_path": str(file)
                    })
            except Exception:
                pass
        sessions.sort(key=lambda s: s.get("session_date", ""), reverse=True)
        return sessions
