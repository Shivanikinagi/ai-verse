# 🔒 Voice Authentication System

> **Secure, real-time, AI-powered voice authentication** — combining deep learning, signal processing, and biometric spoof detection for next‑generation access control.

---

## 🚀 Overview

This project delivers a complete **voice authentication platform** that verifies users by their voice in real time — with spoof detection, low latency, and flexible deployment across web and CLI environments.

Use it for:

* Secure login prototypes
* Research on biometric security
* Enterprise‑grade demos and integrations

---

## 🧠 System Architecture

```
🎤 Microphone Input → 🎧 Audio Processing → 🌐 WebSocket Stream → 🧩 Voice Auth Server → ✅ Decision Engine
        ↑                    ↑                     ↑                     ↑                     ↓
 [Web/CLI Client]   [Preprocessing]       [Real-time Streaming]  [SpeechBrain + ML]     [Access Control]
```

Each component works together to provide sub‑second verification while ensuring robust spoof detection.

---

## 🔁 Feature Flow

### 1️⃣ Audio Capture & Preprocessing

* Capture via **Web Audio API** (web) or **sounddevice** (CLI)
* Convert to **16 kHz mono PCM**, normalize, and package for WebSocket transmission

### 2️⃣ Real-time Communication

* **WebSocket protocol** ensures full‑duplex, low‑latency streaming
* Optimized for responsiveness and scalability

### 3️⃣ Voice Authentication Pipeline

```
Incoming Audio → Speaker Recognition → Spoof Detection → Similarity Comparison → Access Decision
```

#### 🗣️ Speaker Recognition (ECAPA-TDNN)

* Model: `speechbrain/spkrec-ecapa-voxceleb`
* Extracts embeddings and compares via **cosine similarity**

#### 🧪 Spoof Detection (ML‑Based)

* Random Forest classifier with 17 handcrafted audio features (MFCCs, spectral flatness, RMS variance, etc.)
* Flags replay/synthetic voice attacks

#### 🧮 Similarity Comparison

* **Cosine similarity** threshold: 0.80 (configurable)
* ≥ 0.80 → ✅ ACCESS_GRANTED
* < 0.80 → ❌ ACCESS_DENIED

---

## ⚙️ Technology Stack

### 🖥️ Backend

| Component         | Technology       | Purpose                        |
| ----------------- | ---------------- | ------------------------------ |
| Framework         | **FastAPI**      | WebSocket server & REST API    |
| ASGI Server       | **Uvicorn**      | High-performance async runtime |
| Voice Recognition | **SpeechBrain**  | Speaker embedding generation   |
| ML Framework      | **PyTorch**      | Neural network inference       |
| Spoof Detection   | **Scikit-learn** | Random Forest classifier       |
| Audio Processing  | **NumPy**        | Efficient numeric operations   |

### 🌐 Frontend

| Component     | Technology             | Purpose                          |
| ------------- | ---------------------- | -------------------------------- |
| Web Client    | **HTML5 + JavaScript** | Browser-based interface          |
| Audio Capture | **Web Audio API**      | Mic access & preprocessing       |
| WebSocket     | **Native Browser API** | Real-time audio streaming        |
| CLI Client    | **Python**             | Command-line authentication tool |

### 🧩 Dev & Deployment

| Component          | Technology           | Purpose               |
| ------------------ | -------------------- | --------------------- |
| Package Management | **pip**              | Dependency management |
| Model Hosting      | **Hugging Face Hub** | Pre-trained models    |
| Public Access      | **ngrok**            | Tunnel for demos      |
| Runtime            | **Python 3.8+**      | Core environment      |

---

## 🔐 Security Layers

### 1️⃣ Spoof Detection

* Multi-stage ML classifier detects replay and synthetic voice
* Spectral + temporal + heuristic features

### 2️⃣ Voice Biometrics

* One-shot enrollment
* Embedding-based identity representation
* Adjustable similarity thresholds for risk tuning

### 3️⃣ Communication Security

* `wss://` encryption for all audio streams
* Binary audio payloads ensure data integrity

---

## ⚡ Performance

| Metric                       | Description                 | Typical Value |
| ---------------------------- | --------------------------- | ------------- |
| Audio Processing             | Per chunk                   | < 10 ms       |
| Model Inference              | Embedding + Spoof detection | 50–100 ms     |
| Total Round Trip             | End-to-end latency          | < 200 ms      |
| Speaker Recognition Accuracy | Clean audio                 | > 95%         |
| Spoof Detection Accuracy     | Trained attack types        | > 90%         |
| False Acceptance Rate        | Threshold = 0.80            | < 1%          |

---

## 📈 Scalability

* Supports **100+ concurrent users**
* Baseline memory: ~500MB
* CPU usage: ~20% on modern processors
* Easy horizontal scaling with multiple ASGI workers

---

## 🧰 Quickstart

```bash
git clone <REPO_URL>
cd voice-auth-system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Optional:**

```bash
docker build -t voice-auth-server .
docker run -p 8000:8000 voice-auth-server
```

Then visit `http://localhost:8000/docs` for the interactive API.

---

## 🧩 Example WebSocket Messages

**Client → Server:** binary audio chunks (Float32 PCM @16kHz)

**Server → Client:**

```json
{ "event": "partial_result", "similarity": 0.76 }
{ "event": "decision", "result": "ACCESS_GRANTED", "similarity": 0.87 }
{ "event": "spoof_alert", "score": 0.93 }
```

---

## 🧠 Future Enhancements

* Adaptive thresholding using user‑specific statistics
* Transformer-based spoof detection model
* Integration with FIDO2 / MFA backends
* Real‑time dashboard with Grafana

---

## 🤝 Contributing

Pull requests welcome! Please open issues for feedback or improvements. Ensure tests and docs are updated.

---

## 📜 License

Released under **MIT License**. See `LICENSE` for details.

---

### 💡 Maintainers

**VoiceAuth Project Team**
Feel free to reach out for collaborations or research inquiries.
