import os
import time
import base64
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from pathlib import Path

import config
from core.face_tracker import FaceTracker
from core.engagement_analyzer import EngagementAnalyzer
from core.audio_analyzer import AudioAnalyzer
from core.multimodal_fusion import MultimodalFusionEngine
from core.teaching_agent import TeachingAgent
from core.attendance_manager import AttendanceManager
from core.video_processor import VideoProcessor

# Initialize Flask App Gateway
app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = str(config.UPLOADS_DIR)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max file upload

# Initialize Core Services
attendance_mgr = AttendanceManager()
face_tracker = FaceTracker()
face_tracker.set_roster(attendance_mgr.roster)
engagement_analyzer = EngagementAnalyzer()
audio_analyzer = AudioAnalyzer()
fusion_engine = MultimodalFusionEngine()
teaching_agent = TeachingAgent()
video_processor = VideoProcessor(face_tracker, engagement_analyzer)

# Global Session State
active_session = {
    "session_id": f"session_{int(time.time())}",
    "is_active": True,
    "topic": "Introduction to Neural Networks & Classroom Dynamics",
    "start_time": time.time(),
    "last_evaluation_time": time.time(),
    "video_source": "camera"
}

# Start background audio analyzer
audio_analyzer.start_live_audio()


# ==========================================================
# PAGE ROUTES (Frontend Layer)
# ==========================================================

@app.route("/")
def index():
    """Main Live Classroom Attendance & Engagement Dashboard."""
    return render_template("index.html", session=active_session)


@app.route("/upload")
def upload_page():
    """Batch Processing & Video File Upload Dashboard."""
    return render_template("upload.html", session=active_session)


@app.route("/analytics")
def analytics_page():
    """Deep Session Dynamics, Attention Trends & Attendance Reports."""
    saved_sessions = attendance_mgr.list_saved_sessions()
    return render_template("analytics.html", saved_sessions=saved_sessions, session=active_session)


@app.route("/roster")
def roster_page():
    """Student Roster & Face Profile Enrollment Management."""
    return render_template("roster.html", roster=attendance_mgr.roster, session=active_session)


# ==========================================================
# VIDEO STREAMING & INGESTION (Video Input Layer)
# ==========================================================

@app.route("/video_feed")
def video_feed():
    """MJPEG Live Video Stream with Augmented Reality Visual Overlays."""
    if not video_processor.is_camera_active:
        video_processor.start_camera()

    return Response(
        video_processor.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    """
    Browser-direct frame ingestion.
    Allows client-side webcam feed (HTML5 canvas) to send JPEG frames to server.
    Ensures live AI processing even when camera hardware is only on the client machine!
    """
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image")
    audio_level = data.get("audio_level", None)

    if not image_b64:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Decode base64 image
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Invalid image format"}), 400

        now = time.time()
        # Process frame
        annotated_frame, tracks = video_processor.process_single_frame(frame, current_time=now)

        # Process client-sent audio level if available
        if audio_level is not None:
            pseudo_samples = np.array([float(audio_level)] * 512, dtype=np.float32)
            audio_analyzer.process_audio_chunk(pseudo_samples)

        # Compute Visual Aggregates
        visual_data = engagement_analyzer.analyze_classroom_aggregate(tracks)
        audio_data = audio_analyzer.get_current_metrics()

        # Multimodal Fusion
        fusion_metrics = fusion_engine.fuse(visual_data, audio_data, tracks, current_time=now)

        # AI Teaching Suggestions
        suggestion = teaching_agent.evaluate_and_suggest(
            fusion_metrics, visual_data, audio_data, topic_context=active_session.get("topic")
        )

        # Serialize track information
        tracks_serialized = [t.to_dict() for t in tracks]

        # Encode annotated image back to base64
        ret, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8") if ret else None

        return jsonify({
            "status": "success",
            "tracks": tracks_serialized,
            "visual_metrics": visual_data,
            "audio_metrics": audio_data,
            "fusion_metrics": fusion_metrics,
            "new_suggestion": suggestion,
            "annotated_frame": f"data:image/jpeg;base64,{annotated_b64}" if annotated_b64 else None
        })
    except Exception as e:
        return jsonify({"error": f"Frame processing error: {str(e)}"}), 500


# ==========================================================
# REST API GATEWAY (Metrics, Attendance, Suggestions)
# ==========================================================

@app.route("/api/metrics")
def get_metrics():
    """Retrieve current fused multimodal classroom metrics."""
    tracks = video_processor.latest_tracks
    visual_data = engagement_analyzer.analyze_classroom_aggregate(tracks)
    audio_data = audio_analyzer.get_current_metrics()
    fusion_metrics = fusion_engine.fuse(visual_data, audio_data, tracks)
    
    return jsonify({
        "session_id": active_session["session_id"],
        "elapsed_seconds": int(time.time() - active_session["start_time"]),
        "visual_metrics": visual_data,
        "audio_metrics": audio_data,
        "fusion_metrics": fusion_metrics,
        "tracks": [t.to_dict() for t in tracks]
    })


@app.route("/api/attendance")
def get_attendance():
    """Retrieve real-time attendance status based on sustained presence logic."""
    elapsed = time.time() - active_session["start_time"]
    report = attendance_mgr.generate_attendance_report(
        face_tracker.tracks,
        session_elapsed_sec=elapsed,
        session_id=active_session["session_id"]
    )
    return jsonify(report)


@app.route("/api/suggestions")
def get_suggestions():
    """Retrieve latest AI teaching suggestions."""
    recent = teaching_agent.get_recent_suggestions(limit=15)
    return jsonify({"suggestions": recent})


@app.route("/api/session/start", methods=["POST"])
def start_session():
    """Start or restart a classroom monitoring session."""
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "Classroom Session")
    
    active_session["session_id"] = f"session_{int(time.time())}"
    active_session["topic"] = topic
    active_session["start_time"] = time.time()
    active_session["is_active"] = True

    face_tracker.reset_session()
    fusion_engine.reset_session()
    teaching_agent.reset_session()

    return jsonify({
        "status": "started",
        "session_id": active_session["session_id"],
        "topic": topic
    })


@app.route("/api/session/stop", methods=["POST"])
def stop_session():
    """Finalize active session and persist attendance & dynamics report."""
    elapsed = time.time() - active_session["start_time"]
    report = attendance_mgr.generate_attendance_report(
        face_tracker.tracks,
        session_elapsed_sec=elapsed,
        session_id=active_session["session_id"]
    )
    summary = fusion_engine.get_session_summary()
    
    # Save session dynamics JSON
    dynamics_path = config.ANALYTICS_DIR / f"{active_session['session_id']}_dynamics.json"
    try:
        with open(dynamics_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_info": active_session,
                "attendance": report,
                "dynamics": summary
            }, f, indent=2)
    except Exception as e:
        print(f"Error saving session dynamics: {e}")

    active_session["is_active"] = False

    return jsonify({
        "status": "stopped",
        "session_id": active_session["session_id"],
        "attendance": report,
        "dynamics": summary
    })


@app.route("/api/session/export/<session_id>/<file_format>")
def export_session(session_id: str, file_format: str):
    """Download attendance log in CSV or JSON format."""
    if file_format.lower() == "csv":
        filepath = config.ATTENDANCE_LOGS_DIR / f"{session_id}.csv"
        mimetype = "text/csv"
    else:
        filepath = config.ATTENDANCE_LOGS_DIR / f"{session_id}.json"
        mimetype = "application/json"

    if filepath.exists():
        return send_file(str(filepath), as_attachment=True, mimetype=mimetype)
    return jsonify({"error": "Report file not found"}), 404


@app.route("/api/session/details/<session_id>")
def session_details(session_id: str):
    """Retrieve full session dynamics and attendance for report rendering."""
    json_path = config.ATTENDANCE_LOGS_DIR / f"{session_id}.json"
    dynamics_path = config.ANALYTICS_DIR / f"{session_id}_dynamics.json"
    
    if dynamics_path.exists():
        with open(dynamics_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    elif json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return jsonify({"attendance": json.load(f), "dynamics": None})
    return jsonify({"error": "Session not found"}), 404


# ==========================================================
# ROSTER MANAGEMENT ROUTES
# ==========================================================

@app.route("/api/roster", methods=["GET", "POST", "DELETE"])
def manage_roster():
    """List, enroll, or delete students in the roster."""
    if request.method == "GET":
        return jsonify({"roster": attendance_mgr.roster})

    elif request.method == "POST":
        data = request.get_json() or {}
        student_id = data.get("student_id")
        name = data.get("name")
        department = data.get("department", "General")

        if not student_id or not name:
            return jsonify({"error": "Student ID and Name are required"}), 400

        entry = attendance_mgr.enroll_student(student_id, name, department)
        face_tracker.set_roster(attendance_mgr.roster)
        return jsonify({"status": "enrolled", "student": entry})

    elif request.method == "DELETE":
        data = request.get_json() or {}
        student_id = data.get("student_id")
        if not student_id:
            return jsonify({"error": "Student ID required"}), 400

        deleted = attendance_mgr.remove_student(student_id)
        face_tracker.set_roster(attendance_mgr.roster)
        return jsonify({"status": "deleted" if deleted else "not_found"})


# ==========================================================
# BATCH VIDEO FILE UPLOAD & PROCESSING
# ==========================================================

@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """Upload recorded classroom video/audio file and process it offline."""
    if "video_file" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video_file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    save_path = config.UPLOADS_DIR / filename
    file.save(str(save_path))

    # Optional topic tag
    topic = request.form.get("topic", f"Recorded Session ({filename})")
    new_session_id = f"batch_{int(time.time())}"

    # Analyze Audio track using librosa
    audio_report = audio_analyzer.analyze_audio_file(str(save_path))

    # Process Video Frames
    cap = cv2.VideoCapture(str(save_path))
    if not cap.isOpened():
        return jsonify({"error": "Failed to read uploaded video format"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    batch_face_tracker = FaceTracker()
    batch_face_tracker.set_roster(attendance_mgr.roster)
    batch_engagement = EngagementAnalyzer()
    batch_fusion = MultimodalFusionEngine()
    batch_agent = TeachingAgent()

    # Step through frames (sampling 1 frame per second for fast batch turnaround)
    step = max(1, int(fps))
    curr_frame_idx = 0
    simulated_time = 0.0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
        ret, frame = cap.read()
        if not ret or curr_frame_idx >= total_frames:
            break

        current_time = batch_face_tracker.session_start_time + simulated_time
        tracks = batch_face_tracker.detect_and_track(frame, current_time=current_time)

        for track in tracks:
            batch_engagement.analyze_student_engagement(track)

        vis_data = batch_engagement.analyze_classroom_aggregate(tracks)
        
        # Pull corresponding audio second if available
        audio_sub = {"rms": 0.03, "interaction_score": 60.0, "interaction_category": "Lecture"}
        if "timeline" in audio_report and int(simulated_time) < len(audio_report["timeline"]):
            a_item = audio_report["timeline"][int(simulated_time)]
            audio_sub = {
                "rms": a_item["rms"],
                "interaction_score": a_item["interaction_score"],
                "interaction_category": a_item["category"]
            }

        fusion_res = batch_fusion.fuse(vis_data, audio_sub, tracks, current_time=current_time)
        batch_agent.evaluate_and_suggest(fusion_res, vis_data, audio_sub, topic_context=topic)

        curr_frame_idx += step
        simulated_time += 1.0

    cap.release()

    # Generate Final Attendance Report
    final_attendance = attendance_mgr.generate_attendance_report(
        batch_face_tracker.tracks,
        session_elapsed_sec=video_duration,
        session_id=new_session_id
    )
    final_dynamics = batch_fusion.get_session_summary()
    suggestions = batch_agent.get_recent_suggestions(limit=20)

    # Save Batch Results
    batch_result = {
        "session_id": new_session_id,
        "video_filename": filename,
        "video_duration_sec": round(video_duration, 1),
        "topic": topic,
        "attendance": final_attendance,
        "dynamics": final_dynamics,
        "audio_report": audio_report,
        "suggestions": suggestions
    }

    result_path = config.ANALYTICS_DIR / f"{new_session_id}_dynamics.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(batch_result, f, indent=2)

    return jsonify({
        "status": "success",
        "session_id": new_session_id,
        "video_duration_sec": round(video_duration, 1),
        "attendance": final_attendance,
        "dynamics": final_dynamics,
        "audio_report": audio_report,
        "suggestions": suggestions
    })


# ==========================================================
# PRIVACY & CONTROLS TOGGLE
# ==========================================================

@app.route("/api/toggle_privacy", methods=["POST"])
def toggle_privacy():
    """Toggle face anonymization (blurring)."""
    video_processor.anonymize_faces = not video_processor.anonymize_faces
    return jsonify({
        "status": "updated",
        "anonymize_faces": video_processor.anonymize_faces
    })


if __name__ == "__main__":
    print("=" * 65)
    print(" Multimodal Attendance & Engagement Agent Gateway")
    print(f" Running at: http://localhost:{config.PORT}")
    print(" 100% Privacy-Preserving Offline Intelligence Enabled")
    print("=" * 65)
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
