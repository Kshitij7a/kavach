# 🛡️ Project Kavach: Real-Time Deepfake & Scam Detection

Project Kavach is an end-to-end, real-time audio analysis pipeline designed to detect AI-generated deepfake voices and identify malicious scam intents. Built completely in Python, it integrates hardware-level audio capture with a custom machine learning architecture.

## 🚀 Key Features

* **Live Audio Streaming:** Seamlessly captures live audio from the system microphone using `PyAudio`.
* **Energy-Based VAD (Voice Activity Detection):** Implements an RMS-based mathematical threshold to filter out background noise and silence, saving CPU/RAM resources and preventing NLP hallucinations.
* **Deepfake CNN Classifier:** A custom PyTorch-based 1D Convolutional Neural Network trained on extracted MFCC (Mel-frequency cepstral coefficients) features to distinguish between real human voices and AI-generated deepfakes.
* **Scam Intent NLP Engine:** Utilizes OpenAI's `Whisper` model for real-time Speech-to-Text transcription, coupled with a robust keyword and Regex engine to calculate Threat Levels (0% to 100%).
* **Concurrent Orchestration:** Uses Python's `concurrent.futures` to run Deepfake detection and Scam intent analysis in parallel, ensuring zero latency in the main audio thread.

## 🏗️ System Architecture

1. **Audio Streamer:** Captures 16kHz mono audio in chunks.
2. **VAD Gate:** Checks RMS energy; blocks silent chunks.
3. **Feature Extractor:** Converts raw PCM bytes to Numpy arrays and extracts MFCCs.
4. **Dual Inference Engine:** 
   - *Path A:* Passes MFCCs to the trained Deepfake CNN.
   - *Path B:* Passes audio array to Whisper for transcription & Intent Regex.
5. **Orchestrator:** Aggregates scores and logs real-time threat levels.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   
```bash
   git clone [https://github.com/your-username/kavach.git](https://github.com/your-username/kavach.git)
   cd kavach