import math
import numpy as np
from typing import Dict, List, Tuple
import config
from core.face_tracker import StudentTrack


class EngagementAnalyzer:
    """
    Computes visual behavioral engagement scores based on:
    - 3D Head Pose (Pitch, Yaw, Roll)
    - Eye Openness & Drowsiness
    - Motion Velocity & Inactivity/Slump
    """

    def __init__(self):
        pass

    def analyze_student_engagement(self, track: StudentTrack) -> float:
        """
        Evaluate single student engagement score (0 - 100) and update track behavioral flags.
        """
        score = 100.0
        is_attentive = True

        # 1. Yaw Analysis (Looking left/right away from instructional center)
        abs_yaw = abs(track.yaw)
        if abs_yaw > config.HEAD_POSE_YAW_THRESHOLD:
            # Gradual penalty proportional to turning angle
            yaw_penalty = min(35.0, (abs_yaw - config.HEAD_POSE_YAW_THRESHOLD) * 1.5)
            score -= yaw_penalty
            track.distraction_count += 1
            is_attentive = False

        # 2. Pitch Analysis (Looking down at phone/desk or up at ceiling)
        if track.pitch < config.HEAD_POSE_PITCH_DOWN:
            # Looking downward
            pitch_penalty = min(40.0, abs(track.pitch - config.HEAD_POSE_PITCH_DOWN) * 1.8)
            score -= pitch_penalty
            track.distraction_count += 1
            is_attentive = False
        elif track.pitch > config.HEAD_POSE_PITCH_UP:
            # Looking up
            pitch_penalty = min(25.0, (track.pitch - config.HEAD_POSE_PITCH_UP) * 1.2)
            score -= pitch_penalty

        # 3. Eye Openness & Drowsiness Analysis
        if track.eye_openness < config.EYE_ASPECT_RATIO_DROWSY:
            drowsy_penalty = 35.0
            score -= drowsy_penalty
            track.drowsy_count += 1
            is_attentive = False
        elif track.eye_openness < 0.45:
            score -= 15.0

        # 4. Inactivity Slump vs Restless Fidgeting
        if track.inactivity_seconds > config.MAX_INACTIVITY_SEC:
            # Prolonged immobility / head resting on desk
            score -= min(30.0, (track.inactivity_seconds - config.MAX_INACTIVITY_SEC) * 1.0)
        elif track.motion_energy > 25.0:
            # Excessive movement / agitation
            score -= 10.0

        # Clamp score to [5.0, 100.0]
        final_score = max(5.0, min(100.0, score))
        track.current_engagement_score = round(final_score, 1)
        track.is_attentive = is_attentive
        track.engagement_history.append(final_score)
        
        # Keep recent sliding window history
        if len(track.engagement_history) > 120:
            track.engagement_history.pop(0)

        return final_score

    def analyze_classroom_aggregate(self, tracks: List[StudentTrack]) -> dict:
        """
        Aggregate classroom-level visual dynamics across all active participants.
        """
        if not tracks:
            return {
                "active_student_count": 0,
                "average_visual_engagement": 0.0,
                "attentive_count": 0,
                "distracted_count": 0,
                "drowsy_count": 0,
                "restless_count": 0,
                "engagement_distribution": {"high": 0, "medium": 0, "low": 0}
            }

        scores = []
        attentive_count = 0
        distracted_count = 0
        drowsy_count = 0
        restless_count = 0

        distrib = {"high": 0, "medium": 0, "low": 0}  # high >= 70, medium 45-69, low < 45

        for track in tracks:
            s = self.analyze_student_engagement(track)
            scores.append(s)

            if track.is_attentive:
                attentive_count += 1
            else:
                distracted_count += 1

            if track.eye_openness < config.EYE_ASPECT_RATIO_DROWSY:
                drowsy_count += 1

            if track.motion_energy > 20.0:
                restless_count += 1

            if s >= 70.0:
                distrib["high"] += 1
            elif s >= 45.0:
                distrib["medium"] += 1
            else:
                distrib["low"] += 1

        avg_score = float(np.mean(scores)) if scores else 0.0

        return {
            "active_student_count": len(tracks),
            "average_visual_engagement": round(avg_score, 1),
            "attentive_count": attentive_count,
            "distracted_count": distracted_count,
            "drowsy_count": drowsy_count,
            "restless_count": restless_count,
            "engagement_distribution": distrib
        }
