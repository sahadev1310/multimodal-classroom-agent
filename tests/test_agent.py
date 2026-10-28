import unittest
from core.teaching_agent import TeachingAgent


class TestTeachingAgent(unittest.TestCase):
    """Unit tests for AI Teaching Suggestions Agent."""

    def setUp(self):
        self.agent = TeachingAgent()
        self.agent.cooldown_sec = 0.0  # Disable cooldown during unit testing

    def test_attention_drop_suggestion(self):
        """Low CEI or high distraction must trigger PACING / Checkpoint suggestion."""
        fusion = {"cei": 38.0, "engagement_state": "Disengaged / Critical Drop"}
        visual = {"distracted_count": 8, "active_student_count": 10, "drowsy_count": 0, "restless_count": 0}
        audio = {"prolonged_silence_seconds": 10.0, "interaction_category": "Quiet"}

        suggestion = self.agent.evaluate_and_suggest(fusion, visual, audio)
        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion["category"], "PACING")
        self.assertEqual(suggestion["priority"], "high")

    def test_prolonged_silence_suggestion(self):
        """Extended monologue / silence must trigger INTERACTION suggestion."""
        fusion = {"cei": 58.0, "engagement_state": "Moderately Engaged"}
        visual = {"distracted_count": 1, "active_student_count": 10, "drowsy_count": 0, "restless_count": 0}
        audio = {"prolonged_silence_seconds": 220.0, "interaction_category": "Quiet"}

        suggestion = self.agent.evaluate_and_suggest(fusion, visual, audio)
        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion["category"], "INTERACTION")

    def test_high_fatigue_suggestion(self):
        """High drowsiness ratio must trigger FATIGUE suggestion."""
        fusion = {"cei": 55.0, "engagement_state": "Moderately Engaged"}
        visual = {"distracted_count": 0, "active_student_count": 10, "drowsy_count": 5, "restless_count": 0}
        audio = {"prolonged_silence_seconds": 30.0, "interaction_category": "Lecture"}

        suggestion = self.agent.evaluate_and_suggest(fusion, visual, audio)
        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion["category"], "FATIGUE")

    def test_kudos_on_interaction_peak(self):
        """Active discussion + high CEI must trigger KUDOS positive feedback."""
        fusion = {"cei": 86.0, "engagement_state": "Highly Engaged"}
        visual = {"distracted_count": 0, "active_student_count": 10, "drowsy_count": 0, "restless_count": 0}
        audio = {"prolonged_silence_seconds": 0.0, "interaction_category": "Active Discussion"}

        suggestion = self.agent.evaluate_and_suggest(fusion, visual, audio)
        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion["category"], "KUDOS")


if __name__ == "__main__":
    unittest.main()
