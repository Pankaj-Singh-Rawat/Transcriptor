import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import tempfile
import openai
import queue
import threading
import time
import soundfile as sf

# ───────────── API Key ─────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ───────────── Streamlit UI ─────────────
st.set_page_config(page_title="Better Transcriptor", page_icon="🎙️", layout="wide")
st.title("🎙️ Better Transcriptor")
st.write("Click **Start Recording**, speak your answer, then click **Stop Recording** to get the transcription.")

# ───────────── Audio Processor ─────────────
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()

    def recv_audio(self, frame):
        audio = frame.to_ndarray().flatten().astype("float32")
        with self.lock:
            self.audio_buffer = np.concatenate((self.audio_buffer, audio))
        return frame

    def get_audio(self):
        with self.lock:
            buf = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)
        return buf

# ───────────── WebRTC Streamer ─────────────
processor_ref = {"obj": None}
webrtc_ctx = webrtc_streamer(
    key="transcriptor",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True
)

if webrtc_ctx and webrtc_ctx.audio_processor:
    processor: AudioProcessor = webrtc_ctx.audio_processor
    processor_ref["obj"] = processor

    if st.button("🛑 Stop & Transcribe"):
        audio_data = processor.get_audio()
        if len(audio_data) == 0:
            st.error("No audio recorded. Please try again.")
        else:
            # Save to temp WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, audio_data, 16000)
                audio_file = open(tmp.name, "rb")

                st.info("⏳ Transcribing... please wait")
                transcript = openai.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file
                )

                st.success("✅ Transcription:")
                st.write(transcript["text"])
