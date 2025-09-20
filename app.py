import os
import json
import pyaudio
import websockets
import asyncio

API_KEY = os.getenv("OPENAI_API_KEY")  # set this in your terminal before running
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

# Audio settings
RATE = 16000
CHUNK = 1024

async def send_audio(ws):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("🎙️ Listening... Speak into your mic!")

    try:
        while True:
            data = stream.read(CHUNK)
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": data.hex()
            }))
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

async def receive_transcripts(ws):
    async for message in ws:
        event = json.loads(message)
        if event.get("type") == "transcript.delta":
            text = event["delta"]
            print("You said:", text, flush=True)

async def main():
    async with websockets.connect(
        URL,
        extra_headers={
            "Authorization": f"Bearer {API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        },
        ping_interval=20,
        ping_timeout=20
    ) as ws:
        await asyncio.gather(
            send_audio(ws),
            receive_transcripts(ws)
        )

if __name__ == "__main__":
    asyncio.run(main())
