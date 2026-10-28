import time
import numpy as np
from typing import List, Dict, Optional
import config
from core.face_tracker import StudentTrack


class MultimodalFusionEngine:
    """
    Multimodal Temporal Fusion Engine.
    Merges:
    1. Visual behavioral cues (head pose, eye openness, gaze, inactivity)
    2. Motion dynamics (alertness vs restlessness vs slump)
    3. Audio signal features (RMS energy, speech levels, interaction categories)

    Produces:
    - Classroom Engagement Index (CEI) in [0, 100]
    - Anomaly detection: Attention Drops, Interaction Peaks, Fatigue Alerts
    - Continuous time-series logs for real-time visualization and retrospective analysis
    """

    def __init__(self):
        self.session_start_time = time.time()
        self.time_series_history: List[dict] = []
        self.attention_drops: List[dict] = []
        self.interaction_peaks: List[dict] = []
        self.last_recorded_sec = -1
        self.recent_cei_buffer: List[float] = []

    def reset_session(self):
        """Reset historical timeline."""
        self.session_start_time = time.time()
        self.time_series_history.clear()
        self.attention_drops.clear()
        self.interaction_peaks.clear()
        self.last_recorded_sec = -1
        self.recent_cei_buffer.clear()

    def fuse(self, visual_data: dict, audio_data: dict, active_tracks: List[StudentTrack], current_time: Optional[float] = None) -> dict:
        """
        Fuse simultaneous visual and audio modalities into a unified Classroom Engagement Index.
        """
        if current_time is None:
            current_time = time.time()

        elapsed_sec = int(current_time - self.session_start_time)

        # 1. Visual Attention Score (0 - 100)
        v_score = visual_data.get("average_visual_engagement", 70.0)
        if len(active_tracks) == 0:
            v_score = 50.0  # Neutral baseline when no students detected yet

        # 2. Motion Alertness Score (0 - 100)
        # Low motion / slump or extreme commotion is penalized; moderate alertness rewarded
        motion_energies = [t.motion_energy for t in active_tracks] if active_tracks else [5.0]
        avg_motion = float(np.mean(motion_energies))
        if avg_motion < 1.0:
            m_score = 50.0  # Slightly drowsy/inactive
        elif avg_motion <= 15.0:
            m_score = 90.0  # Attentive, controlled motion
        elif avg_motion <= 30.0:
            m_score = 75.0  # Slightly restless
        else:
            m_score = 40.0  # Agitated / disruptive

        # 3. Audio Interaction Score (0 - 100)
        a_score = audio_data.get("interaction_score", 50.0)

        # 4. Multimodal Fusion Calculation
        cei = (
            config.FUSION_WEIGHT_VISUAL_ATTENTION * v_score +
            config.FUSION_WEIGHT_MOVEMENT_ALERTNESS * m_score +
            config.FUSION_WEIGHT_AUDIO_INTERACTION * a_score
        )
        cei = round(max(0.0, min(100.0, cei)), 1)
        self.recent_cei_buffer.append(cei)
        if len(self.recent_cei_buffer) > 30:
            self.recent_cei_buffer.pop(0)

        # Determine engagement qualitative state
        if cei >= 75.0:
            engagement_state = "Highly Engaged"
        elif cei >= 55.0:
            engagement_state = "Moderately Engaged"
        elif cei >= 40.0:
            engagement_state = "Attention Declining"
        else:
            engagement_state = "Disengaged / Critical Drop"

        # Record interval datapoints once every second
        event_tag = None
        if elapsed_sec > self.last_recorded_sec:
            self.last_recorded_sec = elapsed_sec

            # Detect Anomaly: Attention Drop
            if len(self.recent_cei_buffer) >= 5:
                rolling_avg = float(np.mean(self.recent_cei_buffer[:-1]))
                if rolling_avg - cei > 18.0 or cei < config.ALERT_ENGAGEMENT_DROP_THRESHOLD:
                    event_tag = "Attention Drop"
                    drop_record = {
                        "timestamp": elapsed_sec,
                        "time_str": time.strftime("%H:%M:%S", time.gmtime(elapsed_sec)),
                        "cei": cei,
                        "visual_score": round(v_score, 1),
                        "audio_score": round(a_score, 1),
                        "description": f"Sharp attention drop to {cei} (down from {round(rolling_avg, 1)})"
                    }
                    if not self.attention_drops or (elapsed_sec - self.attention_drops[-1]["timestamp"] > 15):
                        self.attention_drops.append(drop_record)

            # Detect Anomaly: Interaction Peak
            if a_score >= 85.0 and v_score >= 65.0:
                event_tag = "Interaction Peak"
                peak_record = {
                    "timestamp": elapsed_sec,
                    "time_str": time.strftime("%H:%M:%S", time.gmtime(elapsed_sec)),
                    "cei": cei,
                    "audio_score": round(a_score, 1),
                    "description": f"High interaction peak detected (audio: {round(a_score, 1)}, CEI: {cei})"
                }
                if not self.interaction_peaks or (elapsed_sec - self.interaction_peaks[-1]["timestamp"] > 20):
                    self.interaction_peaks.append(peak_record)

            # Store in session time series
            datapoint = {
                "second": elapsed_sec,
                "time_str": time.strftime("%H:%M:%S", time.gmtime(elapsed_sec)),
                "cei": cei,
                "visual_score": round(v_score, 1),
                "motion_score": round(m_score, 1),
                "audio_score": round(a_score, 1),
                "audio_rms": audio_data.get("rms", 0.0),
                "audio_category": audio_data.get("interaction_category", "Quiet"),
                "student_count": len(active_tracks),
                "attentive_count": visual_data.get("attentive_count", 0),
                "distracted_count": visual_data.get("distracted_count", 0),
                "event": event_tag
            }
            self.time_series_history.append(datapoint)
            # Retain maximum 3600 seconds (1 hour session) in memory
            if len(self.time_series_history) > 3600:
                self.time_series_history.pop(0)

        return {
            "elapsed_seconds": elapsed_sec,
            "session_time": time.strftime("%H:%M:%S", time.gmtime(elapsed_sec)),
            "cei": cei,
            "engagement_state": engagement_state,
            "visual_score": round(v_score, 1),
            "motion_score": round(m_score, 1),
            "audio_score": round(a_score, 1),
            "audio_category": audio_data.get("interaction_category", "Quiet"),
            "attention_drops_count": len(self.attention_drops),
            "interaction_peaks_count": len(self.interaction_peaks),
            "active_students": len(active_tracks),
            "latest_event": event_tag
        }

    def get_session_summary(self) -> dict:
        """Calculate statistical overview for dashboard reports."""
        if not self.time_series_history:
            return {
                "total_duration_sec": 0,
                "average_cei": 0.0,
                "peak_cei": 0.0,
                "min_cei": 0.0,
                "attention_drops": [],
                "interaction_peaks": [],
                "time_series": []
            }

        ceis = [p["cei"] for p in self.time_series_history]
        return {
            "total_duration_sec": len(self.time_series_history),
            "duration_formatted": time.strftime("%H:%M:%S", time.gmtime(len(self.time_series_history))),
            "average_cei": round(float(np.mean(ceis)), 1),
            "peak_cei": round(float(np.max(ceis)), 1),
            "min_cei": round(float(np.min(ceis)), 1),
            "attention_drops": self.attention_drops,
            "interaction_peaks": self.interaction_peaks,
            "time_series": self.time_series_history
        }
