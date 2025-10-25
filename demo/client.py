import asyncio
import websockets
import sounddevice as sd
import numpy as np

# Use wss://<ngrok-host>/ws when using ngrok; keep ws:// for local testing
# For ngrok, replace with your actual ngrok URL
SERVER_WS = "ws://127.0.0.1:8000/ws"  # Local testing
# Example for ngrok: SERVER_WS = "wss://chiquita-intown-capably.ngrok-free.dev/ws"

SAMPLE_RATE = 16000
DURATION = 3  # seconds per chunk

async def run_client():
    print(f"Connecting to {SERVER_WS} ...")
    try:
        async with websockets.connect(SERVER_WS) as websocket:
            print("Connected. Press Ctrl+C to stop.")
            while True:
                print("Speak now...")
                audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
                sd.wait()
                pcm = audio.flatten()
                await websocket.send(pcm.tobytes())
                try:
                    resp = await websocket.recv()
                    print("Server:", resp)
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by server")
                    break
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_client())
    except KeyboardInterrupt:
        print("\nClient stopped.")