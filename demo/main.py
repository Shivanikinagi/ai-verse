import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import torch
# Try the new SpeechBrain inference API first
try:
    from speechbrain.inference import SpeakerRecognition
except ImportError:
    # Fallback to the old API if the new one is not available
    from speechbrain.pretrained import SpeakerRecognition
from auth_utils import get_similarity
from spoof_detector import spoof_probability
import os

app = FastAPI(title="Live Voice Auth WebSocket")

print("Loading speaker embedding model (speechbrain ECAPA). This may take a few seconds...")

# Handle symlink issue on Windows by setting environment variables
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

# Try to load the model with better error handling
spkrec = None
try:
    print("Attempting to load model via torch.hub (recommended for Windows)...")
    # Use torch.hub.load to avoid HF hub symlink issues on Windows
    spkrec = torch.hub.load(
        'speechbrain/speechbrain', 'spkrec_ecapa_voxceleb',
        source='github', trust_repo=True,  # needed on newer torch
        run_opts={"device": "cpu"}, 
        savedir=os.path.join(os.path.dirname(__file__), "pretrained_models/spkrec_ecapa")
    )
    print("Model loaded successfully via torch.hub!")
except Exception as e:
    print(f"Error loading model via torch.hub: {e}")
    print("Falling back to direct model loading...")
    try:
        # Check if model files exist locally
        model_dir = os.path.join(os.path.dirname(__file__), "pretrained_models/spkrec_ecapa")
        required_files = [
            "hyperparams.yaml",
            "embedding_model.ckpt",
            "mean_var_norm_emb.ckpt",
            "classifier.ckpt",
            "label_encoder.txt"
        ]
        
        # Check if all required files exist
        all_files_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
        
        if os.path.exists(model_dir) and all_files_exist:
            print(f"Model files found in {model_dir}.")
            print("Attempting to load model (this may fail on Windows due to symlink issues)...")
            # Try to load the model with local_files_only parameter
            spkrec = SpeakerRecognition.from_hparams(
                source=model_dir,
                savedir=model_dir,
                run_opts={"device": "cpu"},
                use_auth_token=False,
                local_files_only=True  # Use only local files
            )
            print("Model loaded successfully!")
        else:
            print(f"Model files not found in {model_dir}. To use the real model, please:")
            print("1. Run the setup_model.py script to download the model files")
            print("2. Restart the server")
            print("Using dummy model for testing.")
            spkrec = None
    except Exception as e2:
        print(f"Error loading model via direct method: {e2}")
        print("This is a known issue on Windows with SpeechBrain symlink handling.")
        print("The system will use a dummy model for testing.")
        print("For production use, consider running on Linux or using a Docker container.")
        spkrec = None

# In-memory store for demo: {username: embedding}
# For hackathon demo we support single in-memory user; extend to DB for production.
registered_embedding = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global registered_embedding
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"Received {len(data)} bytes of audio data")
            # interpret incoming bytes as float32 PCM
            pcm = np.frombuffer(data, dtype=np.float32)
            print(f"PCM data shape: {pcm.shape}")
            if pcm.size < 160:  # sanity check
                print("Audio data too short")
                await websocket.send_text("ERR: chunk too short")
                continue

            # get embedding
            if spkrec is not None:
                waveform = torch.from_numpy(pcm).unsqueeze(0)  # [1, time]
                with torch.no_grad():
                    emb = spkrec.encode_batch(waveform).squeeze(0).cpu().numpy()
            else:
                # Dummy embedding for testing
                emb = np.random.rand(192).astype(np.float32)
            
            print(f"Generated embedding shape: {emb.shape}")

            # if first sample, register user (demo behavior)
            if registered_embedding is None:
                registered_embedding = emb
                print("Registered new user embedding")
                await websocket.send_text("REGISTERED: voice sample stored (demo). Speak again to authenticate.")
                continue

            # quick anti-spoof heuristic (0..1)
            spf = spoof_probability(pcm)
            print(f"Spoof probability: {spf}")
            if spf > 0.5:
                await websocket.send_text(f"SPOOF_DETECTED: prob={spf:.2f}")
                continue

            # similarity
            sim = get_similarity(registered_embedding, emb)
            print(f"Similarity: {sim}")
            if sim >= 0.80:
                await websocket.send_text(f"ACCESS_GRANTED: similarity={sim:.3f}")
            else:
                await websocket.send_text(f"ACCESS_DENIED: similarity={sim:.3f}")

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Server error:", e)
        try:
            await websocket.send_text(f"ERR: {e}")
        except:
            pass