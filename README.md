# Multimodal Attendance & Engagement Agent

> **An AI-powered, privacy-preserving classroom intelligence system integrating computer vision, audio signal processing, and agentic pedagogical reasoning.**

---

## 🎯 Executive Overview & Objectives

In educational institutions and corporate training environments, traditional attendance tracking is largely manual or limited to basic biometric card taps that fail to capture meaningful learner engagement, participation context, or session effectiveness. 

The **Multimodal Attendance & Engagement Agent** transforms attendance and session telemetry into continuous, actionable pedagogical insights. Designed to operate **100% offline**, the system ensures strict student privacy compliance while empowering educators with real-time decision support.

### Key Capabilities
- **Automated Attendance Tracking via Sustained Presence Logic**: Replaces error-prone single-frame detections with continuous temporal observation ($\ge 60\%$ sustained presence ratio), eliminating passersby false positives and generating timestamped audit logs.
- **Multimodal Visual & Audio Engagement Fusion**: Blends 3D head pose estimation (Pitch, Yaw, Roll), gaze deviation, drowsiness/inactivity slumping, and acoustic speech energy to compute the dynamic **Classroom Engagement Index (CEI)**.
- **Context-Aware AI Teaching Suggestions Agent**: Operates an offline heuristic reasoner (with optional Google Gemini 2.0 Flash integration) that continuously detects attention dips, conversational lulls, or fatigue, providing instant, non-intrusive pedagogical interventions.
- **Privacy-Preserving Edge Architecture**: All facial feature processing, audio signal analysis, and session logs reside strictly on local storage (`storage/`), with on-the-fly face anonymization (blurring) toggle.

---

## 🏗️ System Architecture

The architecture mirrors the tiered modular layout:

```
[ Frontend Layer ]
├── HTML5 / CSS3 Responsive Glassmorphism Dashboard
├── Chart.js Dynamic Time-Series Telemetry
└── JavaScript Controllers (Dual-mode: Server Feed or Browser Webcam Ingestion)

[ Flask API Gateway & Route Handlers ]
├── /video_feed           (MJPEG Augmented Reality Stream)
├── /api/process_frame    (Browser Canvas Frame Processing)
├── /api/metrics          (Real-time Fused Multimodal Data)
├── /api/attendance       (Sustained Presence Evaluation & Logs)
├── /api/suggestions      (AI Pedagogical Suggestions Feed)
└── /api/upload_video     (Batch Offline Video/Audio Signal Modeling)

[ Core Processing & Multimodal Fusion ]
├── core.face_tracker         (MediaPipe 468-Landmark Mesh + Spatial IoU Tracking)
├── core.engagement_analyzer  (Head Pose Pitch/Yaw/Roll, Drowsiness, Inactivity)
├── core.audio_analyzer       (Librosa / SoundDevice RMS, VAD, Discussion Modeling)
├── core.multimodal_fusion    (Temporal Modality Weighting: CEI = 0.55*V + 0.15*M + 0.30*A)
└── core.teaching_agent       (Pedagogical Decision Engine + Gemini 2.0 Flash Hybrid)

[ Privacy-Preserving Local Storage Layer ]
├── storage/attendance_logs/   (Timestamped CSV and JSON Logs)
├── storage/session_analytics/ (Longitudinal Dynamics & Event Telemetry)
├── storage/roster/            (Class Enrollment Roster)
└── storage/uploads/           (Uploaded Classroom Recordings)
```

---

## ⚡ Mathematical Modeling & Formulations

### 1. Sustained Presence Logic (Attendance Qualification)
Given a session duration $T_{session}$ and an individual student track $i$ observed with continuous streak tolerance:
$$\text{Presence Ratio } R_i = \frac{T_{sustained, i}}{T_{session}}$$

$$\text{Status}(i) = \begin{cases} 
\text{Present}, & R_i \ge 0.60 \text{ and } T_{sustained, i} \ge T_{streak\_min} \\ 
\text{Partial Attendance}, & 0.30 \le R_i < 0.60 \\ 
\text{Absent}, & R_i < 0.30 
\end{cases}$$

### 2. Multimodal Classroom Engagement Index (CEI)
$$CEI_t = w_{v} \cdot E_{visual, t} + w_{m} \cdot M_{alertness, t} + w_{a} \cdot A_{audio, t}$$
Where:
- $w_{v} = 0.55$: Visual attentiveness (head pose center alignment, eye openness).
- $w_{m} = 0.15$: Motion alertness (rewarding controlled motion, penalizing slump or frantic commotion).
- $w_{a} = 0.30$: Audio interaction dynamics (distinguishing quiet monologue from active collaborative discussion).

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### 2. Installation
```powershell
# Navigate to the project directory
cd "C:\Users\vishw\.gemini\antigravity\scratch\multimodal-classroom-agent"

# Install dependencies
py -m pip install -r requirements.txt
```

### 3. Generate Sample Data (Optional)
To immediately populate the dashboard with synthetic classroom session data and a test video:
```powershell
py test_data/generate_sample_session.py
```

### 4. Run Unit Tests
```powershell
py -m unittest discover tests
```

### 5. Launch Application
```powershell
py app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 🖥️ Scenarios Walkthrough

### Scenario 1: Automated Attendance Tracking
- Navigate to **Live Classroom** (`/`).
- If you have a local webcam attached to the server, the MJPEG stream displays bounding boxes, student names, and presence continuity.
- If using a laptop or remote client, click **"Switch to Browser Webcam"** to ingest video frames and audio directly via the browser canvas API.
- Continuous attendance logs are visible in the table and can be exported as **CSV** or **JSON** at any time.

### Scenario 2: Engagement & Session Dynamics Analysis
- The dashboard graphs dynamic CEI fluctuations, highlighting:
  - **Attention Drops**: Triggered when visual focus or CEI declines sharply.
  - **Interaction Peaks**: Highlighted during active conversational discussion.
- Navigate to **Session Analytics** (`/analytics`) to explore longitudinal trends, historical lectures, and anomaly audit logs.

### Scenario 3: AI-Driven Teaching Suggestions
- The AI Teaching Agent cards appear in real time:
  - **PACING**: Prompts quick check-for-understanding questions when attention dips.
  - **INTERACTION**: Suggests Think-Pair-Share activities when monologue exceeds 3 minutes.
  - **FATIGUE**: Advises posture adjustment or a brief stretch when drowsiness spikes.
  - **KUDOS**: Acknowledges vibrant collaborative debates.
- Click **"🔈 Voice Readout: ON"** to hear interventions read aloud using the Web Speech API.

---

## 🛡️ Privacy & Ethical Deployment
1. **Zero External Telemetry**: Runs fully self-contained offline. No webcam images or audio recordings are transmitted to external cloud servers unless an instructor explicitly connects a Gemini API key.
2. **On-the-Fly Anonymization**: The **"🛡️ Face Anonymization"** button blurs all facial regions in real-time.
3. **Auditability**: Attendance decisions are transparently calculated based on verifiable time ratios.
