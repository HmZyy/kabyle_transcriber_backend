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
warnings.filterwarnings("ignore")

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        logger.info(f"Initializing server on {host}:{port}")
        logger.info(f"SSL enabled: {bool(ssl_cert and ssl_key)}")
        logger.info(f"Using device: {self.device}")
    
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
            logger.info(f"Sent transcription to client {self.clients.get(websocket, {}).get('id', 'unknown')}")
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
    
    def load_audio_from_bytes(self, audio_bytes, target_sr=16000):
        """Load audio from bytes"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            audio, sr = librosa.load(temp_path, sr=target_sr)
            
            os.unlink(temp_path)
            
            return audio
        except Exception as e:
            logger.error(f"Error loading audio from bytes: {e}")
            return None
    
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
    
    async def handle_audio_message(self, websocket, message):
        """Handle incoming audio message"""
        try:
            audio_id = message.get("audio_id", str(uuid.uuid4()))
            audio_format = message.get("format", "base64")
            
            await self.send_state(websocket, "processing", "Processing audio...", {"audio_id": audio_id})
            
            # Decode audio data
            if audio_format == "base64":
                audio_bytes = base64.b64decode(message["audio_data"])
            else:
                await self.send_error(websocket, "Unsupported audio format", "UNSUPPORTED_FORMAT")
                return
            
            # Load audio
            audio_data = self.load_audio_from_bytes(audio_bytes)
            if audio_data is None:
                await self.send_error(websocket, "Failed to load audio data", "AUDIO_LOAD_ERROR")
                return
            
            # Transcribe
            transcription = await self.transcribe_audio(audio_data)
            if transcription is None:
                await self.send_error(websocket, "Transcription failed", "TRANSCRIPTION_ERROR")
                return
            
            # Send result
            await self.send_transcription(websocket, transcription, audio_id)
            await self.send_state(websocket, "ready", "Ready for next audio")
            
        except Exception as e:
            logger.error(f"Error handling audio message: {e}")
            await self.send_error(websocket, f"Internal server error: {str(e)}", "INTERNAL_ERROR")
            await self.send_state(websocket, "ready", "Ready for next audio")
    
    async def handle_client(self, websocket, path):
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
            await self.send_state(websocket, "ready", "Connected to Whisper transcription server")
            self.clients[websocket]["state"] = "ready"
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    if message_type == "audio":
                        await self.handle_audio_message(websocket, data)
                    elif message_type == "ping":
                        await self.send_state(websocket, "pong", "Server is alive")
                    elif message_type == "status":
                        current_state = self.clients[websocket]["state"]
                        await self.send_state(websocket, current_state, "Current status")
                    elif message_type == "device_info":
                        print(data)
                    else:
                        await self.send_error(websocket, f"Unknown message type: {message_type}", "UNKNOWN_MESSAGE_TYPE")
                        
                except json.JSONDecodeError:
                    await self.send_error(websocket, "Invalid JSON message", "INVALID_JSON")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.send_error(websocket, f"Error processing message: {str(e)}", "MESSAGE_PROCESSING_ERROR")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Unexpected error with client {client_id}: {e}")
        finally:
            if websocket in self.clients:
                del self.clients[websocket]
            logger.info(f"Cleaned up client {client_id}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        if self.model is None:
            self.load_model()
        
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
    parser = argparse.ArgumentParser(description="Whisper WebSocket Transcription Server with SSL Support")
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
