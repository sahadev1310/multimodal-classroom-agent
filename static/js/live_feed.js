/**
 * Live Multimodal Feed & Real-Time Analytics Controller
 * Handles:
 * - Server MJPEG Stream or Client Browser Webcam with Canvas Ingestion
 * - Real-Time Multimodal Charts (Chart.js)
 * - Dynamic Sustained Presence Table Updates
 * - AI Teaching Suggestion Feed & Web Speech Audio TTS
 */

let clientWebcamStream = null;
let clientWebcamInterval = null;
let useClientWebcam = false;
let engagementChart = null;
let audioContext = null;
let analyserNode = null;
let microphoneSource = null;
let liveAudioRMS = 0.0;
let ttsEnabled = false;
let lastSpokenSuggestionId = null;

document.addEventListener("DOMContentLoaded", () => {
    initChart();
    setupEventListeners();
    startPollingMetrics();
    fetchAttendance();
    fetchSuggestions();
});

// ==========================================================
// REAL-TIME ENGAGEMENT TIME-SERIES CHART (Chart.js)
// ==========================================================
function initChart() {
    const ctx = document.getElementById("engagementChart");
    if (!ctx) return;

    engagementChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Classroom Engagement Index (CEI)",
                    data: [],
                    borderColor: "#06b6d4",
                    backgroundColor: "rgba(6, 182, 212, 0.15)",
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.35,
                    pointRadius: 2,
                    pointHoverRadius: 6
                },
                {
                    label: "Visual Focus (%)",
                    data: [],
                    borderColor: "#10b981",
                    borderWidth: 1.8,
                    borderDash: [4, 4],
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: "Audio Interaction (%)",
                    data: [],
                    borderColor: "#f59e0b",
                    borderWidth: 1.8,
                    borderDash: [2, 2],
                    tension: 0.3,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 350 },
            scales: {
                x: {
                    grid: { color: "rgba(148, 163, 184, 0.08)" },
                    ticks: { color: "#94a3b8", maxTicksLimit: 10 }
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: "rgba(148, 163, 184, 0.08)" },
                    ticks: { color: "#94a3b8" }
                }
            },
            plugins: {
                legend: {
                    labels: { color: "#cbd5e1", font: { size: 11 } }
                },
                tooltip: {
                    backgroundColor: "rgba(15, 23, 42, 0.9)",
                    titleColor: "#f8fafc",
                    bodyColor: "#cbd5e1",
                    borderColor: "rgba(148, 163, 184, 0.2)",
                    borderWidth: 1
                }
            }
        }
    });
}

function updateChart(timeLabel, cei, visual, audio) {
    if (!engagementChart) return;
    const maxDataPoints = 30;

    engagementChart.data.labels.push(timeLabel);
    engagementChart.data.datasets[0].data.push(cei);
    engagementChart.data.datasets[1].data.push(visual);
    engagementChart.data.datasets[2].data.push(audio);

    if (engagementChart.data.labels.length > maxDataPoints) {
        engagementChart.data.labels.shift();
        engagementChart.data.datasets[0].data.shift();
        engagementChart.data.datasets[1].data.shift();
        engagementChart.data.datasets[2].data.shift();
    }
    engagementChart.update("none");
}

// ==========================================================
// POLLING METRICS & REAL-TIME UPDATES
// ==========================================================
function startPollingMetrics() {
    setInterval(async () => {
        if (useClientWebcam) {
            // When client webcam is active, metrics are updated synchronously via process_frame
            return;
        }

        try {
            const res = await fetch("/api/metrics");
            if (!res.ok) return;
            const data = await res.json();
            renderMetrics(data);
        } catch (e) {
            // Quiet network retry
        }
    }, 1000);

    // Refresh attendance and suggestions every 3 seconds
    setInterval(() => {
        fetchAttendance();
        fetchSuggestions();
    }, 3000);
}

function renderMetrics(data) {
    const f = data.fusion_metrics || {};
    const v = data.visual_metrics || {};
    const a = data.audio_metrics || {};

    // Update Top KPI Bar
    const kpiCei = document.getElementById("kpiCei");
    if (kpiCei) kpiCei.innerText = `${f.cei || 0}%`;

    const kpiState = document.getElementById("kpiState");
    if (kpiState) kpiState.innerText = f.engagement_state || "Observing";

    const kpiActive = document.getElementById("kpiActiveStudents");
    if (kpiActive) kpiActive.innerText = f.active_students || 0;

    const kpiAudioCategory = document.getElementById("kpiAudioCategory");
    if (kpiAudioCategory) kpiAudioCategory.innerText = f.audio_category || "Quiet";

    const kpiAttentionDrops = document.getElementById("kpiAttentionDrops");
    if (kpiAttentionDrops) kpiAttentionDrops.innerText = f.attention_drops_count || 0;

    // Update HUD overlay
    const hudTime = document.getElementById("hudSessionTime");
    if (hudTime) hudTime.innerText = f.session_time || "00:00:00";

    const hudStudents = document.getElementById("hudStudentsCount");
    if (hudStudents) hudStudents.innerText = `${f.active_students || 0} Students Detected`;

    // Update Audio Meter Bar
    const meter = document.getElementById("audioMeterFill");
    if (meter) {
        const rms = a.rms || 0;
        const pct = Math.min(100, Math.round(rms * 400));
        meter.style.width = `${pct}%`;
    }

    // Chart update
    const timeLabel = new Date().toLocaleTimeString().split(" ")[0];
    updateChart(timeLabel, f.cei || 0, f.visual_score || 0, f.audio_score || 0);
}

// ==========================================================
// ATTENDANCE & SUSTAINED PRESENCE DATA TABLE
// ==========================================================
async function fetchAttendance() {
    try {
        const res = await fetch("/api/attendance");
        if (!res.ok) return;
        const data = await res.json();
        renderAttendanceTable(data);
    } catch (e) {}
}

function renderAttendanceTable(data) {
    const tbody = document.getElementById("attendanceTableBody");
    if (!tbody) return;

    const records = data.records || [];
    if (records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; padding:1.5rem; color:var(--text-muted)">Awaiting participant face detections...</td></tr>`;
        return;
    }

    let rowsHtml = "";
    records.forEach(r => {
        let statusClass = "absent";
        if (r.status === "Present") statusClass = "present";
        else if (r.status.includes("Partial")) statusClass = "partial";

        rowsHtml += `
            <tr>
                <td><strong>${r.student_id}</strong></td>
                <td>${r.name}</td>
                <td><span class="status-badge ${statusClass}">${r.status}</span></td>
                <td>${r.sustained_duration_sec}s</td>
                <td>
                    <div style="display:flex; align-items:center; gap:0.5rem">
                        <span>${r.sustained_ratio_pct}%</span>
                        <div class="progress-bar-container" style="width:70px">
                            <div class="progress-bar-fill" style="width:${Math.min(100, r.sustained_ratio_pct)}%; background:${r.sustained_ratio_pct >= 60 ? 'var(--accent-emerald)' : (r.sustained_ratio_pct >= 30 ? 'var(--accent-amber)' : 'var(--accent-rose)')}"></div>
                        </div>
                    </div>
                </td>
                <td><strong>${r.avg_engagement_pct}%</strong></td>
                <td>
                    <span title="Distractions" style="color:var(--accent-amber)">⚠️ ${r.distraction_count}</span> | 
                    <span title="Drowsy Events" style="color:var(--accent-rose)">💤 ${r.drowsy_count}</span>
                </td>
            </tr>
        `;
    });

    tbody.innerHTML = rowsHtml;

    // Update attendance rate KPI
    const kpiAttRate = document.getElementById("kpiAttendanceRate");
    if (kpiAttRate) kpiAttRate.innerText = `${data.attendance_rate_pct || 0}%`;
}

// ==========================================================
// AI TEACHING SUGGESTIONS FEED & SPEECH SYNTHESIS
// ==========================================================
async function fetchSuggestions() {
    try {
        const res = await fetch("/api/suggestions");
        if (!res.ok) return;
        const data = await res.json();
        renderSuggestions(data.suggestions || []);
    } catch (e) {}
}

function renderSuggestions(suggestions) {
    const container = document.getElementById("suggestionsContainer");
    if (!container) return;

    if (suggestions.length === 0) {
        container.innerHTML = `
            <div style="text-align:center; padding:2rem; color:var(--text-muted)">
                <p>AI Teaching Assistant is actively monitoring classroom dynamics...</p>
                <small>Suggestions will dynamically appear here when engagement shifts occur.</small>
            </div>
        `;
        return;
    }

    let cardsHtml = "";
    suggestions.slice(-8).reverse().forEach(s => {
        let catClass = (s.category || "pacing").toLowerCase();
        cardsHtml += `
            <div class="suggestion-card ${catClass}">
                <div class="suggestion-header">
                    <span class="suggestion-tag">${s.category}</span>
                    <span class="suggestion-time">${s.time_formatted}</span>
                </div>
                <div class="suggestion-title">${s.title}</div>
                <div class="suggestion-body">${s.suggestion}</div>
                <div class="suggestion-source">
                    <span>⚡ Source: ${s.source}</span>
                    <button class="btn btn-secondary btn-sm" style="margin-left:auto; padding:2px 8px; font-size:11px" onclick="speakText('${escapeQuotes(s.suggestion)}')">🔊 Read</button>
                </div>
            </div>
        `;
    });

    container.innerHTML = cardsHtml;

    // Auto TTS readout for latest high-priority recommendation
    if (ttsEnabled && suggestions.length > 0) {
        const latest = suggestions[suggestions.length - 1];
        if (latest.timestamp !== lastSpokenSuggestionId && (latest.priority === "high" || latest.priority === "critical")) {
            lastSpokenSuggestionId = latest.timestamp;
            speakText(latest.suggestion);
        }
    }
}

function escapeQuotes(str) {
    return (str || "").replace(/'/g, "\\'").replace(/"/g, "&quot;");
}

function speakText(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
}

// ==========================================================
// CLIENT WEBCAM DUAL-MODE CONTROLLER
// ==========================================================
async function toggleClientWebcam() {
    const videoElem = document.getElementById("liveVideo");
    const imgElem = document.getElementById("serverVideoFeed");
    const toggleBtn = document.getElementById("toggleCameraBtn");

    if (!useClientWebcam) {
        try {
            // Request browser camera & microphone
            clientWebcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: true
            });

            videoElem.srcObject = clientWebcamStream;
            videoElem.style.display = "block";
            if (imgElem) imgElem.style.display = "none";

            // Init Web Audio Analyzer
            initClientAudio(clientWebcamStream);

            useClientWebcam = true;
            toggleBtn.innerHTML = "📷 Use Server Camera";
            toggleBtn.classList.remove("btn-secondary");
            toggleBtn.classList.add("btn-primary");

            // Setup canvas capture loop
            const canvas = document.createElement("canvas");
            canvas.width = 640;
            canvas.height = 480;
            const ctx = canvas.getContext("2d");

            clientWebcamInterval = setInterval(async () => {
                if (!useClientWebcam || videoElem.readyState < 2) return;
                ctx.drawImage(videoElem, 0, 0, 640, 480);
                const b64 = canvas.toDataURL("image/jpeg", 0.7);

                try {
                    const res = await fetch("/api/process_frame", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            image: b64,
                            audio_level: liveAudioRMS
                        })
                    });
                    if (!res.ok) return;
                    const data = await res.json();
                    renderMetrics(data);
                    if (data.new_suggestion) {
                        fetchSuggestions();
                    }
                } catch (e) {}
            }, 650); // 1.5 fps capture for optimal browser-to-server throughput

        } catch (err) {
            alert("Could not access browser webcam/microphone: " + err.message);
        }
    } else {
        // Switch back to server camera stream
        if (clientWebcamStream) {
            clientWebcamStream.getTracks().forEach(t => t.stop());
            clientWebcamStream = null;
        }
        if (clientWebcamInterval) {
            clearInterval(clientWebcamInterval);
            clientWebcamInterval = null;
        }
        videoElem.style.display = "none";
        if (imgElem) {
            imgElem.style.display = "block";
            imgElem.src = "/video_feed?t=" + new Date().getTime();
        }

        useClientWebcam = false;
        toggleBtn.innerHTML = "💻 Switch to Browser Webcam";
        toggleBtn.classList.remove("btn-primary");
        toggleBtn.classList.add("btn-secondary");
    }
}

function initClientAudio(stream) {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 512;
        microphoneSource = audioContext.createMediaStreamSource(stream);
        microphoneSource.connect(analyserNode);

        const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
        setInterval(() => {
            if (!analyserNode) return;
            analyserNode.getByteTimeDomainData(dataArray);
            let sumSq = 0;
            for (let i = 0; i < dataArray.length; i++) {
                const norm = (dataArray[i] - 128) / 128.0;
                sumSq += norm * norm;
            }
            liveAudioRMS = Math.sqrt(sumSq / dataArray.length);
        }, 200);
    } catch (e) {
        console.warn("Client Web Audio API error:", e);
    }
}

// ==========================================================
// SESSION CONTROLS & PRIVACY TOGGLE
// ==========================================================
function setupEventListeners() {
    const toggleCamBtn = document.getElementById("toggleCameraBtn");
    if (toggleCamBtn) {
        toggleCamBtn.addEventListener("click", toggleClientWebcam);
    }

    const privacyBtn = document.getElementById("togglePrivacyBtn");
    if (privacyBtn) {
        privacyBtn.addEventListener("click", async () => {
            const res = await fetch("/api/toggle_privacy", { method: "POST" });
            const data = await res.json();
            if (data.anonymize_faces) {
                privacyBtn.classList.remove("btn-secondary");
                privacyBtn.classList.add("btn-success");
                privacyBtn.innerHTML = "🛡️ Face Anonymization: ACTIVE";
            } else {
                privacyBtn.classList.remove("btn-success");
                privacyBtn.classList.add("btn-secondary");
                privacyBtn.innerHTML = "🛡️ Face Anonymization: OFF";
            }
        });
    }

    const ttsToggleBtn = document.getElementById("toggleTtsBtn");
    if (ttsToggleBtn) {
        ttsToggleBtn.addEventListener("click", () => {
            ttsEnabled = !ttsEnabled;
            ttsToggleBtn.innerHTML = ttsEnabled ? "🔊 Voice Readout: ON" : "🔈 Voice Readout: OFF";
            ttsToggleBtn.classList.toggle("btn-primary", ttsEnabled);
            ttsToggleBtn.classList.toggle("btn-secondary", !ttsEnabled);
        });
    }

    const stopSessionBtn = document.getElementById("stopSessionBtn");
    if (stopSessionBtn) {
        stopSessionBtn.addEventListener("click", async () => {
            if (confirm("End current session and finalize attendance logs?")) {
                const res = await fetch("/api/session/stop", { method: "POST" });
                const data = await res.json();
                alert(`Session ${data.session_id} finalized and saved successfully!`);
                window.location.href = "/analytics";
            }
        });
    }
}
