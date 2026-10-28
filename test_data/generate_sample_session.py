import os
import json
import time
import math
import cv2
import numpy as np
from pathlib import Path

# Adjust path to import core config
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from core.face_tracker import StudentTrack
from core.attendance_manager import AttendanceManager
from core.multimodal_fusion import MultimodalFusionEngine
from core.teaching_agent import TeachingAgent


def generate_synthetic_session():
    """Generate a realistic 20-minute classroom session dataset for instant analytics exploration."""
    session_id = "sample_lecture_session_01"
    session_duration_sec = 1200  # 20 minutes (1200 seconds)
    
    print(f"Generating synthetic session: {session_id} ({session_duration_sec}s)...")

    mgr = AttendanceManager()
    fusion = MultimodalFusionEngine()
    agent = TeachingAgent()
    agent.cooldown_sec = 60.0

    # 1. Simulate Student Tracks
    students_data = [
        {"id": "STU-001", "name": "Aarav Sharma", "sustained_pct": 0.95, "avg_eng": 88.0},
        {"id": "STU-002", "name": "Priya Patel", "sustained_pct": 0.92, "avg_eng": 84.0},
        {"id": "STU-003", "name": "Rohan Iyer", "sustained_pct": 0.85, "avg_eng": 72.0},
        {"id": "STU-004", "name": "Ananya Sen", "sustained_pct": 0.90, "avg_eng": 81.0},
        {"id": "STU-005", "name": "Vikram Rao", "sustained_pct": 0.78, "avg_eng": 65.0},
        {"id": "STU-006", "name": "Maya Nambiar", "sustained_pct": 0.64, "avg_eng": 59.0},
        {"id": "STU-007", "name": "Karthik Verma", "sustained_pct": 0.45, "avg_eng": 48.0},  # Partial
        {"id": "STU-008", "name": "Sneha Kulkarni", "sustained_pct": 0.15, "avg_eng": 35.0},  # Absent (<30%)
    ]

    tracks = {}
    now = time.time()
    for s in students_data:
        t = StudentTrack(s["id"], name=s["name"], bbox=(50, 50, 100, 100))
        t.sustained_seconds = session_duration_sec * s["sustained_pct"]
        t.first_seen = now - session_duration_sec
        t.last_seen = now
        t.current_engagement_score = s["avg_eng"]
        t.engagement_history = [s["avg_eng"] + float(np.random.normal(0, 5)) for _ in range(30)]
        t.distraction_count = int((100 - s["avg_eng"]) / 5)
        t.drowsy_count = int(t.distraction_count / 3)
        tracks[s["id"]] = t

    # 2. Simulate Time-Series Trajectory (with real-world attention drop & interaction peak)
    time_series = []
    base_time = now - session_duration_sec

    for sec in range(0, session_duration_sec, 5):  # 5-second intervals
        minute = sec / 60.0
        
        # Natural classroom attention curve:
        # 0-5 mins: Warmup / Focus (75-85)
        # 5-12 mins: Steady Lecture (70-75)
        # 12-15 mins: Attention Drop / Dip (down to 42)
        # 15-18 mins: Interactive Activity / Discussion Peak (88-92)
        # 18-20 mins: Synthesis / Wrapup (75-80)
        if minute < 5.0:
            v_score = 80.0 + math.sin(minute) * 5.0
            a_score = 60.0
            a_cat = "Lecture"
            rms = 0.04
        elif minute < 12.0:
            v_score = 72.0 + math.cos(minute) * 4.0
            a_score = 55.0
            a_cat = "Lecture"
            rms = 0.035
        elif minute < 15.0:
            # Attention Drop
            v_score = 42.0 + math.sin(minute * 2) * 4.0
            a_score = 25.0
            a_cat = "Quiet"
            rms = 0.012
        elif minute < 18.0:
            # Interaction Peak
            v_score = 86.0 + math.sin(minute) * 3.0
            a_score = 92.0
            a_cat = "Active Discussion"
            rms = 0.12
        else:
            v_score = 76.0 + math.cos(minute) * 3.0
            a_score = 65.0
            a_cat = "Lecture"
            rms = 0.05

        v_score = max(10.0, min(100.0, v_score + float(np.random.normal(0, 2))))
        a_score = max(10.0, min(100.0, a_score + float(np.random.normal(0, 3))))
        m_score = 80.0 if minute >= 15 else 70.0

        cei = round(0.55 * v_score + 0.15 * m_score + 0.30 * a_score, 1)

        event = None
        if 13.0 <= minute <= 13.5:
            event = "Attention Drop"
        elif 16.0 <= minute <= 16.5:
            event = "Interaction Peak"

        time_series.append({
            "second": sec,
            "time_str": time.strftime("%H:%M:%S", time.gmtime(sec)),
            "cei": cei,
            "visual_score": round(v_score, 1),
            "motion_score": round(m_score, 1),
            "audio_score": round(a_score, 1),
            "audio_rms": round(rms, 4),
            "audio_category": a_cat,
            "student_count": 8,
            "attentive_count": 6 if cei > 65 else 2,
            "distracted_count": 2 if cei > 65 else 6,
            "event": event
        })

    # 3. Generate Attendance Report
    attendance_report = mgr.generate_attendance_report(
        tracks,
        session_elapsed_sec=session_duration_sec,
        session_id=session_id
    )

    # 4. Save Session Dynamics JSON
    dynamics_summary = {
        "total_duration_sec": session_duration_sec,
        "duration_formatted": "00:20:00",
        "average_cei": round(float(np.mean([p["cei"] for p in time_series])), 1),
        "peak_cei": round(float(np.max([p["cei"] for p in time_series])), 1),
        "min_cei": round(float(np.min([p["cei"] for p in time_series])), 1),
        "attention_drops": [
            {
                "timestamp": 780,
                "time_str": "00:13:00",
                "cei": 41.5,
                "description": "Sharp attention decline to 41.5% during dense conceptual derivation."
            }
        ],
        "interaction_peaks": [
            {
                "timestamp": 960,
                "time_str": "00:16:00",
                "cei": 88.4,
                "description": "High collaborative discussion peak during student pair-programming prompt."
            }
        ],
        "time_series": time_series
    }

    dynamics_file = config.ANALYTICS_DIR / f"{session_id}_dynamics.json"
    with open(dynamics_file, "w", encoding="utf-8") as f:
        json.dump({
            "session_info": {"session_id": session_id, "topic": "Deep Learning Fundamentals", "duration": 1200},
            "attendance": attendance_report,
            "dynamics": dynamics_summary
        }, f, indent=2)

    print(f"Sample session generated successfully at: {dynamics_file}")
    print(f"Attendance records saved: {len(attendance_report['records'])}")
    print(f"Present: {attendance_report['present_count']}, Partial: {attendance_report['partial_count']}, Absent: {attendance_report['absent_count']}")


def generate_sample_video():
    """Create a short 5-second synthetic classroom video file with simulated moving faces."""
    video_path = config.UPLOADS_DIR / "sample_classroom_lecture.mp4"
    if video_path.exists():
        print(f"Sample video already exists: {video_path}")
        return

    print("Generating synthetic classroom video file for batch testing...")
    width, height = 640, 480
    fps = 20
    duration_sec = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    # Pre-draw background
    for frame_idx in range(fps * duration_sec):
        frame = np.full((height, width, 3), (25, 20, 20), dtype=np.uint8)

        # Draw Classroom Board
        cv2.rectangle(frame, (40, 30), (600, 150), (45, 60, 45), -1)
        cv2.rectangle(frame, (40, 30), (600, 150), (120, 150, 120), 2)
        cv2.putText(frame, "LECTURE: MULTIMODAL AI AGENTS", (70, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Draw 3 Simulated Student Faces with subtle motion
        t = frame_idx / fps
        faces = [
            {"x": int(120 + 10 * math.sin(t * 2)), "y": 280, "name": "Aarav"},
            {"x": int(320 + 8 * math.cos(t * 1.5)), "y": 270, "name": "Priya"},
            {"x": int(500 + 12 * math.sin(t)), "y": 290, "name": "Rohan"}
        ]

        for f in faces:
            # Face oval / circle
            cv2.circle(frame, (f["x"], f["y"]), 35, (180, 210, 240), -1)
            cv2.circle(frame, (f["x"], f["y"]), 35, (100, 140, 180), 2)
            # Eyes
            cv2.circle(frame, (f["x"] - 12, f["y"] - 8), 4, (40, 40, 40), -1)
            cv2.circle(frame, (f["x"] + 12, f["y"] - 8), 4, (40, 40, 40), -1)
            # Smile
            cv2.ellipse(frame, (f["x"], f["y"] + 10), (12, 6), 0, 0, 180, (40, 40, 40), 2)
            # Torso
            cv2.ellipse(frame, (f["x"], f["y"] + 80), (45, 40), 0, 0, 180, (80, 120, 180), -1)

        out.write(frame)

    out.release()
    print(f"Sample video created successfully at: {video_path}")


if __name__ == "__main__":
    generate_synthetic_session()
    generate_sample_video()
