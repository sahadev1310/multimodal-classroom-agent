import time
import math
import threading
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import config

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False


class AudioAnalyzer:
    """
    Multimodal Audio Processing Engine.
    Processes live microphone feeds or recorded audio files to compute:
    - RMS Energy (volume/amplitude)
    - Voice Activity & Speech Presence
    - Classroom Interaction Levels (Silence, Lecture, Discussion, Commotion)
    - Audio Interaction Score (0 - 100)
    """

    def __init__(self, sample_rate: int = config.AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.chunk_size = config.AUDIO_CHUNK_SIZE
        self.is_listening = False
        self.audio_thread: Optional[threading.Thread] = None
        self.stream: Optional[object] = None
        
        # Real-time state metrics
        self.lock = threading.Lock()
        self.current_rms = 0.0
        self.current_zcr = 0.0
        self.speech_energy_ratio = 0.0
        self.interaction_score = 50.0  # Default neutral
        self.interaction_category = "Neutral"
        self.recent_energy_history = []
        self.last_speech_time = time.time()
        self.prolonged_silence_seconds = 0.0

    def start_live_audio(self):
        """Start non-blocking background microphone listener if sounddevice is available."""
        if not SOUNDDEVICE_AVAILABLE:
            print("[AudioAnalyzer] sounddevice not available. Operating in simulated/passive mode.")
            return

        if self.is_listening:
            return

        self.is_listening = True
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            )
            self.stream.start()
            print("[AudioAnalyzer] Live microphone stream started successfully.")
        except Exception as e:
            print(f"[AudioAnalyzer] Could not start microphone stream: {e}. Falling back to passive mode.")
            self.is_listening = False

    def stop_live_audio(self):
        """Stop background audio stream."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.is_listening = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Live audio chunk callback."""
        if status:
            pass
        # indata is float32 numpy array
        samples = indata[:, 0]
        self.process_audio_chunk(samples)

    def process_audio_chunk(self, samples: np.ndarray) -> dict:
        """
        Analyze an array of mono audio samples.
        Computes RMS, ZCR, interaction score, and classification.
        """
        if len(samples) == 0:
            return self.get_current_metrics()

        # Compute Root Mean Square (RMS) Energy
        rms = float(np.sqrt(np.mean(samples ** 2)))
        
        # Zero Crossing Rate (ZCR)
        zero_crossings = np.nonzero(np.diff(samples > 0))[0]
        zcr = float(len(zero_crossings) / len(samples)) if len(samples) > 0 else 0.0

        # Frequency energy in speech band (85Hz - 3000Hz)
        speech_presence = False
        if rms > config.AUDIO_RMS_SILENCE_THRESHOLD:
            speech_presence = True

        now = time.time()
        with self.lock:
            self.current_rms = rms
            self.current_zcr = zcr
            self.recent_energy_history.append(rms)
            if len(self.recent_energy_history) > 60:
                self.recent_energy_history.pop(0)

            if speech_presence:
                self.last_speech_time = now
                self.prolonged_silence_seconds = 0.0
            else:
                self.prolonged_silence_seconds = now - self.last_speech_time

            # Interaction Level Classification & Scoring
            # Score scaled to [0 - 100]
            if rms < config.AUDIO_RMS_SILENCE_THRESHOLD:
                # Silence / Focused Quiet Reading
                self.interaction_category = "Quiet / Silence"
                self.interaction_score = 30.0
            elif rms < config.AUDIO_RMS_DISCUSSION_THRESHOLD:
                # Moderate speech: Teacher lecture or single student answer
                self.interaction_category = "Lecture / Monologue"
                self.interaction_score = 65.0
            elif rms < 0.22:
                # High interaction / Active group discussion / Q&A
                self.interaction_category = "Active Discussion"
                self.interaction_score = 90.0
            else:
                # Excessive commotion or loud noise
                self.interaction_category = "High Commotion"
                self.interaction_score = 45.0  # Penalized for distraction

        return self.get_current_metrics()

    def analyze_audio_file(self, file_path: str) -> dict:
        """
        Process an uploaded audio or video file's audio track using librosa.
        Returns time-series metrics over the entire duration.
        """
        if not LIBROSA_AVAILABLE:
            return {"error": "librosa library is not available"}

        try:
            # Load audio file (downsampled to 16kHz mono)
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))

            # Frame-level RMS energy
            frame_length = 2048
            hop_length = 512
            rms_series = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            times = librosa.times_like(rms_series, sr=sr, hop_length=hop_length)

            # Summaries
            mean_rms = float(np.mean(rms_series))
            max_rms = float(np.max(rms_series))
            
            # Interaction timeline (downsampled to 1-second intervals)
            timeline = []
            sec_step = int(sr / hop_length)
            for sec_idx, i in enumerate(range(0, len(rms_series), sec_step)):
                sec_rms = float(np.mean(rms_series[i:i+sec_step]))
                if sec_rms < config.AUDIO_RMS_SILENCE_THRESHOLD:
                    cat = "Quiet"
                    score = 30.0
                elif sec_rms < config.AUDIO_RMS_DISCUSSION_THRESHOLD:
                    cat = "Lecture"
                    score = 65.0
                elif sec_rms < 0.22:
                    cat = "Discussion"
                    score = 90.0
                else:
                    cat = "Commotion"
                    score = 45.0

                timeline.append({
                    "second": sec_idx,
                    "rms": round(sec_rms, 4),
                    "category": cat,
                    "interaction_score": score
                })

            overall_discussion_ratio = sum(1 for item in timeline if item["category"] == "Discussion") / max(1, len(timeline))
            overall_silence_ratio = sum(1 for item in timeline if item["category"] == "Quiet") / max(1, len(timeline))

            return {
                "duration_seconds": round(duration, 1),
                "mean_rms": round(mean_rms, 4),
                "max_rms": round(max_rms, 4),
                "discussion_ratio": round(overall_discussion_ratio * 100, 1),
                "silence_ratio": round(overall_silence_ratio * 100, 1),
                "timeline": timeline
            }
        except Exception as e:
            return {"error": f"Failed to analyze audio file: {str(e)}"}

    def get_current_metrics(self) -> dict:
        """Thread-safe snapshot of live audio state."""
        with self.lock:
            return {
                "rms": round(self.current_rms, 4),
                "zcr": round(self.current_zcr, 4),
                "interaction_score": round(self.interaction_score, 1),
                "interaction_category": self.interaction_category,
                "prolonged_silence_seconds": round(self.prolonged_silence_seconds, 1)
            }
