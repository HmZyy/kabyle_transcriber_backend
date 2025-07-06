#!/usr/bin/env python3
"""
Whisper Kabyle Audio Transcription WebSocket Backend with SSL Support
Real-time audio transcription server using WebSockets with WSS support
"""
import asyncio
import websockets
import json
import base64
import tempfile
import os
import sys
import uuid
import ssl
from datetime import datetime
from pathlib import Path
import torch
import librosa
import numpy as np
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
)
import argparse
import logging
import warnings
import mimetypes
import subprocess
from typing import Dict, List, Optional
warnings.filterwarnings("ignore")

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UploadSession:
    """Track file upload sessions"""
    def __init__(self, upload_id: str, file_name: str, file_size: int, total_chunks: int, chunk_size: int):
        self.upload_id = upload_id
        self.file_name = file_name
        self.file_size = file_size
        self.total_chunks = total_chunks
        self.chunk_size = chunk_size
        self.chunks: Dict[int, bytes] = {}
        self.received_chunks = 0
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.temp_file_path: Optional[str] = None
    
    def add_chunk(self, chunk_index: int, chunk_data: bytes) -> bool:
        """Add a chunk to the session"""
        if chunk_index in self.chunks:
            logger.warning(f"Duplicate chunk {chunk_index} for upload {self.upload_id}")
            return False
        
        self.chunks[chunk_index] = chunk_data
        self.received_chunks += 1
        self.last_activity = datetime.now()
        
        logger.info(f"Upload {self.upload_id}: Received chunk {chunk_index + 1}/{self.total_chunks} ({len(chunk_data)} bytes)")
        return True
    
    def is_complete(self) -> bool:
        """Check if all chunks have been received"""
        return self.received_chunks == self.total_chunks
    
    def get_progress(self) -> float:
        """Get upload progress as percentage"""
        return (self.received_chunks / self.total_chunks) * 100 if self.total_chunks > 0 else 0
    
    def reassemble_file(self) -> bytes:
        """Reassemble chunks into complete file"""
        if not self.is_complete():
            raise ValueError("Cannot reassemble incomplete upload")
        
        # Sort chunks by index and concatenate
        sorted_chunks = [self.chunks[i] for i in sorted(self.chunks.keys())]
        file_data = b''.join(sorted_chunks)
        
        # Verify file size
        if len(file_data) != self.file_size:
            logger.warning(f"File size mismatch: expected {self.file_size}, got {len(file_data)}")
        
        return file_data
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
                logger.info(f"Cleaned up temporary file: {self.temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {self.temp_file_path}: {e}")

class WhisperWebSocketServer:
    def __init__(self, checkpoint_path="./checkpoint", host="0.0.0.0", port=16391, ssl_cert=None, ssl_key=None):
        """
        Initialize the Whisper WebSocket server
        
        Args:
            checkpoint_path (str): Path to the checkpoint directory
            host (str): Server host
            port (int): Server port
            ssl_cert (str): Path to SSL certificate file
            ssl_key (str): Path to SSL private key file
        """
        self.checkpoint_path = checkpoint_path
        self.host = host
        self.port = port
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.clients = {}  # Track connected clients and their states
        self.upload_sessions: Dict[str, UploadSession] = {}  # Track active upload sessions
        
        logger.info(f"Initializing server on {host}:{port}")
        logger.info(f"SSL enabled: {bool(ssl_cert and ssl_key)}")
        logger.info(f"Using device: {self.device}")
        
        # Check for ffmpeg availability for MP3 support
        self.ffmpeg_available = self.check_ffmpeg()
        if self.ffmpeg_available:
            logger.info("FFmpeg detected - MP3 support available")
        else:
            logger.warning("FFmpeg not found - MP3 support may be limited")
    
    def check_ffmpeg(self):
        """Check if ffmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    async def cleanup_old_uploads(self):
        """Clean up old/stale upload sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                current_time = datetime.now()
                
                expired_sessions = []
                for upload_id, session in self.upload_sessions.items():
                    # Remove sessions older than 1 hour or inactive for 10 minutes
                    age = (current_time - session.created_at).total_seconds()
                    inactive = (current_time - session.last_activity).total_seconds()
                    
                    if age > 3600 or inactive > 600:  # 1 hour total or 10 minutes inactive
                        expired_sessions.append(upload_id)
                
                for upload_id in expired_sessions:
                    session = self.upload_sessions.pop(upload_id, None)
                    if session:
                        session.cleanup()
                        logger.info(f"Cleaned up expired upload session: {upload_id}")
                        
            except Exception as e:
                logger.error(f"Error in upload cleanup task: {e}")
    
    def create_ssl_context(self):
        """Create SSL context for WSS"""
        if not (self.ssl_cert and self.ssl_key):
            return None
            
        try:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.ssl_cert, self.ssl_key)
            
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
            
            logger.info("SSL context created successfully")
            return ssl_context
        except Exception as e:
            logger.error(f"Failed to create SSL context: {e}")
            return None
    
    def load_model(self):
        """Load the fine-tuned Whisper model and processor"""
        try:
            logger.info(f"Loading model from {self.checkpoint_path}...")
            
            logger.info("Loading processor from base whisper-small model...")
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            
            logger.info("Loading fine-tuned model weights...")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.checkpoint_path,
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            if hasattr(self.model.config, 'forced_decoder_ids'):
                self.model.config.forced_decoder_ids = None
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            try:
                logger.info("Trying alternative loading method...")
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    "openai/whisper-small",
                    torch_dtype=torch.float32
                )
                
                checkpoint_file = os.path.join(self.checkpoint_path, "model.safetensors")
                if os.path.exists(checkpoint_file) and SAFETENSORS_AVAILABLE:
                    logger.info("Loading fine-tuned weights...")
                    state_dict = load_file(checkpoint_file)
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info("Fine-tuned weights loaded successfully!")
                else:
                    logger.warning("Could not load fine-tuned weights, using base model")
                
                self.model.to(self.device)
                self.model.eval()
                
                if hasattr(self.model.config, 'forced_decoder_ids'):
                    self.model.config.forced_decoder_ids = None
                
            except Exception as e2:
                logger.error(f"Alternative loading method failed: {e2}")
                raise e2
    
    async def send_state(self, websocket, state, message="", data=None):
        """Send state message to client"""
        response = {
            "type": "state",
            "state": state,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        if data:
            response["data"] = data
        
        try:
            await websocket.send(json.dumps(response))
            logger.info(f"Sent state '{state}' to client {self.clients.get(websocket, {}).get('id', 'unknown')}")
        except Exception as e:
            logger.error(f"Error sending state: {e}")
    
    async def send_transcription(self, websocket, transcription, audio_id=None):
        """Send transcription result to client"""
        response = {
            "type": "transcription",
            "transcription": transcription,
            "timestamp": datetime.now().isoformat(),
            "audio_id": audio_id
        }
        
        try:
            await websocket.send(json.dumps(response))
            logger.info(f"Sent transcription to client {self.clients.get(websocket, {}).get('id', 'unknown')}: {transcription[:100]}...")
        except Exception as e:
            logger.error(f"Error sending transcription: {e}")
    
    async def send_error(self, websocket, error_message, error_code=None):
        """Send error message to client"""
        response = {
            "type": "error",
            "message": error_message,
            "code": error_code,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await websocket.send(json.dumps(response))
            logger.error(f"Sent error to client: {error_message}")
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
    
    async def send_upload_progress(self, websocket, upload_id, progress, message=""):
        """Send upload progress update to client"""
        response = {
            "type": "upload_progress",
            "upload_id": upload_id,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await websocket.send(json.dumps(response))
        except Exception as e:
            logger.error(f"Error sending upload progress: {e}")
    
    async def send_upload_complete(self, websocket, upload_id, message="Upload completed successfully"):
        """Send upload completion confirmation to client"""
        response = {
            "type": "upload_complete",
            "upload_id": upload_id,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await websocket.send(json.dumps(response))
            logger.info(f"Sent upload complete notification for {upload_id}")
        except Exception as e:
            logger.error(f"Error sending upload complete: {e}")
    
    def detect_audio_format(self, audio_bytes):
        """Detect audio format from file headers"""
        if len(audio_bytes) < 12:
            return None
            
        # Check for MP3 (ID3 tag or MP3 frame sync)
        if audio_bytes[:3] == b'ID3' or (len(audio_bytes) > 1 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0):
            return 'mp3'
        
        # Check for WAV (RIFF header)
        if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
            return 'wav'
        
        # Check for OGG
        if audio_bytes[:4] == b'OggS':
            return 'ogg'
        
        # Check for FLAC
        if audio_bytes[:4] == b'fLaC':
            return 'flac'
        
        # Check for M4A/AAC
        if len(audio_bytes) > 8 and audio_bytes[4:8] == b'ftyp':
            return 'm4a'
        
        return None
    
    def load_audio_from_bytes(self, audio_bytes, file_format=None, target_sr=16000):
        """Load audio from bytes with proper format detection"""
        temp_file = None
        try:
            # Detect format if not provided
            if not file_format:
                file_format = self.detect_audio_format(audio_bytes)
                if not file_format:
                    logger.warning("Could not detect audio format, assuming MP3")
                    file_format = 'mp3'
            
            logger.info(f"Processing audio file with format: {file_format}")
            
            # Create temporary file with correct extension
            suffix = f'.{file_format}'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            # Load audio with librosa
            try:
                # For MP3 files, we might need to be more explicit about the backend
                if file_format == 'mp3':
                    # Try with different backends
                    audio = None
                    sr = None
                    
                    # Try soundfile first (usually works with ffmpeg)
                    try:
                        audio, sr = librosa.load(temp_path, sr=target_sr, res_type='kaiser_fast')
                        logger.info(f"Successfully loaded MP3 with librosa (soundfile backend)")
                    except Exception as e:
                        logger.warning(f"Soundfile backend failed: {e}")
                    
                    # If that fails and ffmpeg is available, try converting first
                    if audio is None and self.ffmpeg_available:
                        try:
                            # Convert MP3 to WAV using ffmpeg
                            wav_path = temp_path.replace('.mp3', '_converted.wav')
                            cmd = [
                                'ffmpeg', '-i', temp_path, 
                                '-ar', str(target_sr), 
                                '-ac', '1',  # mono
                                '-f', 'wav',
                                '-y',  # overwrite
                                wav_path
                            ]
                            
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                            if result.returncode == 0:
                                audio, sr = librosa.load(wav_path, sr=target_sr)
                                os.unlink(wav_path)  # Clean up converted file
                                logger.info("Successfully converted MP3 to WAV with ffmpeg")
                            else:
                                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                        except Exception as e:
                            logger.error(f"FFmpeg conversion error: {e}")
                    
                    if audio is None:
                        raise Exception("All MP3 loading methods failed")
                else:
                    # For other formats, use librosa directly
                    audio, sr = librosa.load(temp_path, sr=target_sr)
                
                logger.info(f"Audio loaded successfully: {len(audio)} samples at {sr}Hz")
                
                # Validate audio
                if len(audio) == 0:
                    raise Exception("Loaded audio is empty")
                
                # Normalize audio to prevent clipping
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio)) * 0.9
                
                return audio
                
            except Exception as e:
                logger.error(f"Error loading audio with librosa: {e}")
                raise e
            
        except Exception as e:
            logger.error(f"Error loading audio from bytes: {e}")
            return None
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"Could not delete temporary file {temp_path}: {e}")
    
    async def transcribe_audio(self, audio_data):
        """Transcribe audio data"""
        try:
            chunk_size = 30 * 16000  # 30 seconds
            
            if len(audio_data) <= chunk_size:
                transcription = await self.transcribe_chunk(audio_data)
                return transcription
            else:
                transcriptions = []
                overlap = 1 * 16000  # 1 second overlap
                start = 0
                
                logger.info(f"Processing long audio file in chunks: {len(audio_data) / 16000:.1f} seconds")
                
                while start < len(audio_data):
                    end = min(start + chunk_size, len(audio_data))
                    chunk = audio_data[start:end]
                    
                    if len(chunk) < 16000:  # Less than 1 second
                        chunk = np.pad(chunk, (0, 16000 - len(chunk)))
                    
                    transcription = await self.transcribe_chunk(chunk)
                    if transcription:
                        transcriptions.append(transcription)
                    
                    start = end - overlap if end < len(audio_data) else end
                
                return " ".join(transcriptions)
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None
    
    async def transcribe_chunk(self, audio_chunk):
        """Transcribe a single audio chunk"""
        try:
            inputs = self.processor(
                audio_chunk,
                sampling_rate=16000,
                return_tensors="pt"
            )
            
            input_features = inputs.input_features.to(self.device, dtype=self.model.dtype)
            
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,
                    num_beams=5,
                    do_sample=False,
                    temperature=0.0,
                    use_cache=True,
                    task="transcribe",
                    language=None,
                    forced_decoder_ids=None,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error during chunk transcription: {e}")
            return ""
    
    async def handle_upload_start(self, websocket, message):
        """Handle upload start message"""
        try:
            upload_id = message.get("upload_id")
            file_name = message.get("file_name")
            file_size = message.get("file_size")
            total_chunks = message.get("total_chunks")
            chunk_size = message.get("chunk_size")
            
            if not all([upload_id, file_name, file_size, total_chunks, chunk_size]):
                await self.send_error(websocket, "Missing required upload parameters", "MISSING_UPLOAD_PARAMS")
                return
            
            # Validate file type
            allowed_extensions = ['.wav', '.mp3']
            file_extension = Path(file_name).suffix.lower()
            if file_extension not in allowed_extensions:
                await self.send_error(websocket, f"Unsupported file type: {file_extension}. Only .wav and .mp3 files are supported.", "UNSUPPORTED_FILE_TYPE")
                return
            
            # Check if upload already exists
            if upload_id in self.upload_sessions:
                logger.warning(f"Upload session {upload_id} already exists, replacing")
                old_session = self.upload_sessions[upload_id]
                old_session.cleanup()
            
            # Create new upload session
            session = UploadSession(upload_id, file_name, file_size, total_chunks, chunk_size)
            self.upload_sessions[upload_id] = session
            
            logger.info(f"Started upload session: {upload_id} - {file_name} ({file_size} bytes, {total_chunks} chunks)")
            
            await self.send_upload_progress(websocket, upload_id, 0, f"Upload started for {file_name}")
            
        except Exception as e:
            logger.error(f"Error handling upload start: {e}")
            await self.send_error(websocket, f"Failed to start upload: {str(e)}", "UPLOAD_START_ERROR")
    
    async def handle_upload_chunk(self, websocket, message):
        """Handle upload chunk message"""
        try:
            upload_id = message.get("upload_id")
            chunk_index = message.get("chunk_index")
            chunk_data_b64 = message.get("chunk_data")
            is_final_chunk = message.get("is_final_chunk", False)
            
            if not all([upload_id is not None, chunk_index is not None, chunk_data_b64]):
                await self.send_error(websocket, "Missing chunk parameters", "MISSING_CHUNK_PARAMS")
                return
            
            # Get upload session
            session = self.upload_sessions.get(upload_id)
            if not session:
                await self.send_error(websocket, f"Upload session not found: {upload_id}", "UPLOAD_SESSION_NOT_FOUND")
                return
            
            # Decode chunk data
            try:
                chunk_data = base64.b64decode(chunk_data_b64)
            except Exception as e:
                await self.send_error(websocket, f"Failed to decode chunk data: {str(e)}", "CHUNK_DECODE_ERROR")
                return
            
            # Add chunk to session
            success = session.add_chunk(chunk_index, chunk_data)
            if not success:
                await self.send_error(websocket, f"Failed to add chunk {chunk_index}", "CHUNK_ADD_ERROR")
                return
            
            # Send progress update
            progress = session.get_progress()
            await self.send_upload_progress(websocket, upload_id, progress, 
                                          f"Received chunk {chunk_index + 1}/{session.total_chunks}")
            
        except Exception as e:
            logger.error(f"Error handling upload chunk: {e}")
            await self.send_error(websocket, f"Failed to process chunk: {str(e)}", "UPLOAD_CHUNK_ERROR")
    
    async def handle_upload_complete(self, websocket, message):
        """Handle upload complete message"""
        try:
            upload_id = message.get("upload_id")
            file_name = message.get("file_name")
            
            if not upload_id:
                await self.send_error(websocket, "Missing upload ID", "MISSING_UPLOAD_ID")
                return
            
            # Get upload session
            session = self.upload_sessions.get(upload_id)
            if not session:
                await self.send_error(websocket, f"Upload session not found: {upload_id}", "UPLOAD_SESSION_NOT_FOUND")
                return
            
            # Check if upload is complete
            if not session.is_complete():
                await self.send_error(websocket, 
                                    f"Upload incomplete: {session.received_chunks}/{session.total_chunks} chunks received", 
                                    "UPLOAD_INCOMPLETE")
                return
            
            logger.info(f"Processing completed upload: {upload_id} - {session.file_name}")
            
            # Reassemble file
            try:
                file_data = session.reassemble_file()
                logger.info(f"Reassembled file: {len(file_data)} bytes")
            except Exception as e:
                await self.send_error(websocket, f"Failed to reassemble file: {str(e)}", "FILE_REASSEMBLE_ERROR")
                return
            
            # Process the uploaded file for transcription
            await self.send_upload_complete(websocket, upload_id, "File upload completed, starting transcription...")
            
            # Create audio message for processing
            audio_message = {
                "audio_data": base64.b64encode(file_data).decode('utf-8'),
                "format": "base64",
                "audio_id": upload_id,
                "file_name": session.file_name,
                "file_size": len(file_data),
                "is_upload": True
            }
            
            # Process the audio
            await self.handle_audio_message(websocket, audio_message)
            
            # Clean up upload session
            session.cleanup()
            del self.upload_sessions[upload_id]
            logger.info(f"Cleaned up upload session: {upload_id}")
            
        except Exception as e:
            logger.error(f"Error handling upload complete: {e}")
            await self.send_error(websocket, f"Failed to complete upload: {str(e)}", "UPLOAD_COMPLETE_ERROR")
    
    async def handle_upload_error(self, websocket, message):
        """Handle upload error message"""
        try:
            upload_id = message.get("upload_id")
            error_msg = message.get("error", "Unknown upload error")
            
            if upload_id and upload_id in self.upload_sessions:
                session = self.upload_sessions[upload_id]
                session.cleanup()
                del self.upload_sessions[upload_id]
                logger.info(f"Cleaned up failed upload session: {upload_id}")
            
            logger.warning(f"Upload error reported by client: {upload_id} - {error_msg}")
            
        except Exception as e:
            logger.error(f"Error handling upload error: {e}")
    
    async def handle_audio_message(self, websocket, message):
        """Handle incoming audio message"""
        try:
            audio_id = message.get("audio_id", str(uuid.uuid4()))
            audio_format = message.get("format", "base64")
            file_name = message.get("file_name", "")
            file_size = message.get("file_size", 0)
            is_upload = message.get("is_upload", False)
            
            logger.info(f"Processing audio: ID={audio_id}, Format={audio_format}, Upload={is_upload}, File={file_name}, Size={file_size}")
            
            await self.send_state(websocket, "processing", "Processing audio...", {"audio_id": audio_id})
            
            # Decode audio data
            if audio_format == "base64":
                try:
                    audio_bytes = base64.b64decode(message["audio_data"])
                    logger.info(f"Decoded {len(audio_bytes)} bytes of audio data")
                except Exception as e:
                    await self.send_error(websocket, f"Failed to decode base64 audio: {str(e)}", "BASE64_DECODE_ERROR")
                    return
            else:
                await self.send_error(websocket, "Unsupported audio format", "UNSUPPORTED_FORMAT")
                return
            
            # Detect file format for uploads
            detected_format = None
            if is_upload and file_name:
                # Try to get format from filename extension
                ext = file_name.lower().split('.')[-1] if '.' in file_name else None
                if ext in ['mp3', 'wav', 'ogg', 'flac', 'm4a']:
                    detected_format = ext
            
            # Load audio
            await self.send_state(websocket, "loading", "Loading audio data...")
            audio_data = self.load_audio_from_bytes(audio_bytes, detected_format)
            if audio_data is None:
                await self.send_error(websocket, "Failed to load audio data", "AUDIO_LOAD_ERROR")
                return
            
            logger.info(f"Audio loaded successfully: {len(audio_data)} samples")
            
            # Transcribe
            await self.send_state(websocket, "transcribing", "Transcribing audio...")
            transcription = await self.transcribe_audio(audio_data)
            if transcription is None:
                await self.send_error(websocket, "Transcription failed", "TRANSCRIPTION_ERROR")
                return
            
            logger.info(f"Transcription completed: {len(transcription)} characters")
            
            # Send result
            await self.send_transcription(websocket, transcription, audio_id)
            await self.send_state(websocket, "ready", "Ready for next audio")
            
        except Exception as e:
            logger.error(f"Error handling audio message: {e}")
            await self.send_error(websocket, f"Internal server error: {str(e)}", "INTERNAL_ERROR")
            await self.send_state(websocket, "ready", "Ready for next audio")
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        client_id = str(uuid.uuid4())
        client_info = {
            "id": client_id,
            "connected_at": datetime.now(),
            "state": "connecting"
        }
        self.clients[websocket] = client_info
        
        logger.info(f"New client connected: {client_id}")
        
        try:
            await self.send_state(websocket, "ready", "")
            self.clients[websocket]["state"] = "ready"
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    if message_type == "audio":
                        await self.handle_audio_message(websocket, data)
                    elif message_type == "upload_start":
                        await self.handle_upload_start(websocket, data)
                    elif message_type == "upload_chunk":
                        await self.handle_upload_chunk(websocket, data)
                    elif message_type == "upload_complete":
                        await self.handle_upload_complete(websocket, data)
                    elif message_type == "upload_error":
                        await self.handle_upload_error(websocket, data)
                    elif message_type == "ping":
                        await self.send_state(websocket, "pong", "Server is alive")
                    elif message_type == "status":
                        current_state = self.clients[websocket]["state"]
                        await self.send_state(websocket, current_state, "Current status")
                    elif message_type == "device_info":
                        logger.info(f"Device info from {client_id}: {data}")
                    else:
                        await self.send_error(websocket, f"Unknown message type: {message_type}", "UNKNOWN_MESSAGE_TYPE")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    await self.send_error(websocket, "Invalid JSON message", "INVALID_JSON")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.send_error(websocket, f"Error processing message: {str(e)}", "MESSAGE_PROCESSING_ERROR")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Unexpected error with client {client_id}: {e}")
        finally:
            # Clean up client upload sessions
            expired_sessions = []
            for upload_id, session in self.upload_sessions.items():
                # This is a simple cleanup - in a production system you might want to track which client owns which upload
                pass  # For now, we rely on the periodic cleanup task
            
            if websocket in self.clients:
                del self.clients[websocket]
            logger.info(f"Cleaned up client {client_id}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        if self.model is None:
            self.load_model()
        
        # Start the upload cleanup task now that we have an event loop
        asyncio.create_task(self.cleanup_old_uploads())
        
        ssl_context = self.create_ssl_context()
        protocol = "wss" if ssl_context else "ws"
        
        logger.info(f"Starting WebSocket server ({protocol}) on {self.host}:{self.port}")
        
        async with websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            ssl=ssl_context
        ):
            logger.info(f"WebSocket server started successfully on {protocol}://{self.host}:{self.port}")
            logger.info("Waiting for connections...")
            await asyncio.Future()  # Run forever

def main():
    parser = argparse.ArgumentParser(description="Whisper WebSocket Transcription Server with SSL Support and File Upload")
    parser.add_argument("--checkpoint", default="./checkpoint", help="Path to model checkpoint directory")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=16391, help="Server port (default: 16391)")
    parser.add_argument("--ssl-cert", help="Path to SSL certificate file (.crt or .pem)")
    parser.add_argument("--ssl-key", help="Path to SSL private key file (.key)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate checkpoint directory
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint directory not found: {args.checkpoint}")
        sys.exit(1)
    
    # Validate SSL certificates if provided
    if args.ssl_cert or args.ssl_key:
        if not (args.ssl_cert and args.ssl_key):
            logger.error("Both --ssl-cert and --ssl-key must be provided for SSL support")
            sys.exit(1)
        
        if not os.path.exists(args.ssl_cert):
            logger.error(f"SSL certificate file not found: {args.ssl_cert}")
            sys.exit(1)
        
        if not os.path.exists(args.ssl_key):
            logger.error(f"SSL private key file not found: {args.ssl_key}")
            sys.exit(1)
    
    server = WhisperWebSocketServer(
        args.checkpoint, 
        args.host, 
        args.port, 
        args.ssl_cert, 
        args.ssl_key
    )
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
