import asyncio
import websockets
import numpy as np

async def test_client():
    uri = "wss://chiquita-intown-capably.ngrok-free.dev/ws"
    print(f"Connecting to {uri}")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # Send a small test message
            test_data = np.random.rand(16000).astype(np.float32)  # 1 second of random audio
            print(f"Sending {len(test_data)} samples")
            await websocket.send(test_data.tobytes())
            
            # Wait for response
            response = await websocket.recv()
            print(f"Received: {response}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_client())