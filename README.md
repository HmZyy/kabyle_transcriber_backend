# Whisper Kabyle Audio Transcription WebSocket Server

This project provides a real-time WebSocket (WSS) server for streaming audio transcription using a fine-tuned [Whisper](https://huggingface.co/openai/whisper-small) model. It supports SSL for secure communication and is designed to transcribe Kabyle (or any supported) language audio sent from WebSocket clients.

---

## 🔧 Features

- Real-time transcription over WebSocket
- WSS (WebSocket Secure) support using SSL certificates
- Fine-tuned Whisper model support (`safetensors` or `pytorch_model.bin`)
- JSON-based client communication
- Efficient 30s chunk-based audio transcription with 1s overlap
- Includes connection state updates and error reporting

---

## 🧪 Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/HmZyy/kabyle_transcriber_backend.git
cd kabyle_transcriber_backend
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> Ensure you have `ffmpeg` installed for `librosa` audio loading.

---

## 📦 Install `cloudflared` (Optional: Expose localhost to public HTTPS)

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
```

Then run:

```bash
cloudflared tunnel --url https://localhost:16391
```

---

## 🚀 Run the Server

```bash
python3 main.py --host 127.0.0.1 --port 16391 --checkpoint ./checkpoint
```

## 🧠 WebSocket Message Format

### Audio Message

```json
{
  "type": "audio",
  "audio_id": "optional-uuid",
  "format": "base64",
  "audio_data": "<base64_encoded_wav_data>"
}
```

### Ping Message

```json
{ "type": "ping" }
```

---

## 📥 Response Format

### Transcription

```json
{
  "type": "transcription",
  "transcription": "your transcribed text",
  "timestamp": "ISO 8601 timestamp",
  "audio_id": "optional-uuid"
}
```

### State Update

```json
{
  "type": "state",
  "state": "processing | ready | pong | error",
  "message": "status message",
  "timestamp": "ISO 8601 timestamp"
}
```

### Error

```json
{
  "type": "error",
  "message": "error message",
  "code": "error code",
  "timestamp": "ISO 8601 timestamp"
}
```

---

## 🧠 Model Checkpoint

Place your fine-tuned Whisper model in the `./checkpoint/` directory. Supported formats:

- `pytorch_model.bin`
- `model.safetensors` (preferred)

The server automatically tries both.

---

## 📜 License

MIT License © 2025
