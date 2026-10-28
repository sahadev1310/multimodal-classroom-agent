import os
import time
from typing import List, Dict, Optional
import config

try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False


class TeachingSuggestion:
    """Represents an actionable pedagogical suggestion generated for the instructor."""

    def __init__(self, category: str, title: str, suggestion: str, priority: str = "medium", source: str = "Rule Engine"):
        self.category = category  # "PACING", "INTERACTION", "ALERT", "FATIGUE", "KUDOS"
        self.title = title
        self.suggestion = suggestion
        self.priority = priority  # "low", "medium", "high", "critical"
        self.source = source      # "Local Rule Engine" or "Gemini 2.0 Flash"
        self.timestamp = time.time()
        self.time_formatted = time.strftime("%H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "title": self.title,
            "suggestion": self.suggestion,
            "priority": self.priority,
            "source": self.source,
            "timestamp": self.timestamp,
            "time_formatted": self.time_formatted
        }


class TeachingAgent:
    """
    AI-Driven Classroom Intelligence & Teaching Suggestion Agent.
    Operates offline by default using expert pedagogical rules,
    with seamless Gemini 2.0 Flash integration when available.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.suggestions_history: List[TeachingSuggestion] = []
        self.last_suggestion_time = 0.0
        self.cooldown_sec = 20.0  # Prevent overwhelming instructor with too frequent alerts
        self.gemini_client = None

        if self.api_key and GENAI_AVAILABLE:
            try:
                self.gemini_client = genai.Client(api_key=self.api_key)
                print("[TeachingAgent] Google Gemini 2.0 Flash AI Service Layer initialized.")
            except Exception as e:
                print(f"[TeachingAgent] Gemini client init warning: {e}. Running in local mode.")

    def reset_session(self):
        """Reset suggestion history."""
        self.suggestions_history.clear()
        self.last_suggestion_time = 0.0

    def evaluate_and_suggest(
        self,
        fusion_metrics: dict,
        visual_metrics: dict,
        audio_metrics: dict,
        topic_context: Optional[str] = None
    ) -> Optional[dict]:
        """
        Continuously evaluate multimodal dynamics and generate actionable recommendations.
        """
        now = time.time()
        if now - self.last_suggestion_time < self.cooldown_sec:
            return None  # In cooldown

        cei = fusion_metrics.get("cei", 70.0)
        audio_silence_sec = audio_metrics.get("prolonged_silence_seconds", 0.0)
        distracted_ratio = (
            visual_metrics.get("distracted_count", 0) / max(1, visual_metrics.get("active_student_count", 1))
        )
        drowsy_ratio = (
            visual_metrics.get("drowsy_count", 0) / max(1, visual_metrics.get("active_student_count", 1))
        )
        restless_ratio = (
            visual_metrics.get("restless_count", 0) / max(1, visual_metrics.get("active_student_count", 1))
        )
        audio_cat = audio_metrics.get("interaction_category", "Quiet")

        suggestion: Optional[TeachingSuggestion] = None

        # 1. Critical Attention Drop (< 45 or sharp decline)
        if cei < config.ALERT_ENGAGEMENT_DROP_THRESHOLD or distracted_ratio >= 0.40:
            suggestion = TeachingSuggestion(
                category="PACING",
                title="Attention Dip Detected",
                suggestion=f"Classroom engagement has dropped to {cei}%. Over {int(distracted_ratio*100)}% of participants are looking away or slouching. Pause your monologue and ask a quick checkpoint question or cold-call an engaging reflection prompt.",
                priority="high",
                source="Local Rule Engine"
            )

        # 2. Prolonged Silence / One-way Monologue (> 3 minutes)
        elif audio_silence_sec >= config.ALERT_PROLONGED_SILENCE_SEC or (audio_cat == "Quiet" and cei < 60.0):
            suggestion = TeachingSuggestion(
                category="INTERACTION",
                title="Low Audio Participation",
                suggestion=f"Continuous silence or one-way lecture detected for over {int(audio_silence_sec//60)} minutes. Spark participation with a 2-minute Think-Pair-Share exercise or an open-ended debate question.",
                priority="medium",
                source="Local Rule Engine"
            )

        # 3. High Drowsiness / Slump Rate (> 30%)
        elif drowsy_ratio >= config.ALERT_HIGH_DROWSINESS_RATIO:
            suggestion = TeachingSuggestion(
                category="FATIGUE",
                title="Learner Fatigue Observed",
                suggestion=f"Approximately {int(drowsy_ratio*100)}% of students exhibit signs of drowsiness or downward head tilt. Introduce a brief physical posture adjustment, change speaking tone, or conduct a rapid live poll.",
                priority="high",
                source="Local Rule Engine"
            )

        # 4. High Restlessness (> 40%)
        elif restless_ratio >= config.ALERT_HIGH_RESTLESSNESS_RATIO:
            suggestion = TeachingSuggestion(
                category="ALERT",
                title="Restlessness & Fidgeting Rising",
                suggestion=f"Significant motion jitter detected across {int(restless_ratio*100)}% of learners. Pacing may feel sluggish or over-dense. Transition to an interactive slide or problem demonstration.",
                priority="medium",
                source="Local Rule Engine"
            )

        # 5. Positive Reinforcement (Interaction Peak: High Audio Discussion + High Visual Focus)
        elif audio_cat == "Active Discussion" and cei >= 80.0:
            if not self._has_recent_category("KUDOS", 120.0):
                suggestion = TeachingSuggestion(
                    category="KUDOS",
                    title="Optimal Collaborative Engagement",
                    suggestion=f"Classroom discussion is peaking at {cei}% engagement index with high conversational reciprocity. Great dynamic—allow students 2 more minutes to synthesize findings before closing the activity.",
                    priority="low",
                    source="Local Rule Engine"
                )

        # If Gemini AI Service Layer is active and available, enrich or synthesize deeper prompt
        if suggestion and self.gemini_client and not config.ENABLE_OFFLINE_MODE:
            suggestion = self._enhance_with_gemini(suggestion, fusion_metrics, topic_context)

        if suggestion:
            self.last_suggestion_time = now
            self.suggestions_history.append(suggestion)
            if len(self.suggestions_history) > 50:
                self.suggestions_history.pop(0)
            return suggestion.to_dict()

        return None

    def _enhance_with_gemini(
        self,
        base_suggestion: TeachingSuggestion,
        fusion_metrics: dict,
        topic_context: Optional[str]
    ) -> TeachingSuggestion:
        """Query Gemini 2.0 Flash for creative pedagogical adaptation."""
        try:
            prompt = (
                f"You are an AI teaching assistant observing a live university/training session. "
                f"Topic: {topic_context or 'General Lecture'}. "
                f"Classroom Metrics: Engagement Index: {fusion_metrics.get('cei')}%, State: {fusion_metrics.get('engagement_state')}. "
                f"The local system observed: {base_suggestion.suggestion}. "
                f"Provide a concise, 2-sentence actionable teaching intervention and a sample question the instructor can immediately ask."
            )
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            if response.text:
                base_suggestion.suggestion = response.text.strip()
                base_suggestion.source = "Google Gemini 2.0 Flash"
        except Exception as e:
            # Fall back to local rule suggestion
            pass
        return base_suggestion

    def _has_recent_category(self, category: str, window_sec: float) -> bool:
        """Check if a suggestion of category was made within the last window_sec."""
        now = time.time()
        for s in reversed(self.suggestions_history):
            if s.category == category and (now - s.timestamp) < window_sec:
                return True
        return False

    def get_recent_suggestions(self, limit: int = 10) -> List[dict]:
        """Return the most recent suggestion cards."""
        return [s.to_dict() for s in self.suggestions_history[-limit:]]
