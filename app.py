import streamlit as st
from faster_whisper import WhisperModel
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import queue

# ─────────────── SETTINGS ───────────────
WHISPER_SIZE = "small"
DEVICE = "cpu"

# ─────────────── INIT MODEL ───────────────
@st.cache_resource
def load_model():
    with st.spinner("🔄 Loading Whisper model..."):
        return WhisperModel(
            WHISPER_SIZE,
            device=DEVICE,
            compute_type="int8",
            cpu_threads=2,
            download_root="./models"
        )

model = load_model()

# ─────────────── AUDIO PROCESSOR ───────────────
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_queue = queue.Queue()

    def recv_audio(self, frame):
        # Incoming audio frames are sent here
        audio_array = frame.to_ndarray().flatten().astype("float32")
        self.audio_queue.put(audio_array)
        return frame

# ─────────────── STREAMLIT UI ───────────────
st.set_page_config(page_title="Live Transcriber", page_icon="🎙️", layout="wide")
st.title("🎙️ Real-Time Speech Transcriber")

webrtc_ctx = webrtc_streamer(
    key="speech-transcriber",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx and webrtc_ctx.audio_processor:
    processor: AudioProcessor = webrtc_ctx.audio_processor
    transcript_box = st.empty()
    full_text = ""

    # Process audio in background
    while True:
        try:
            audio_chunk = processor.audio_queue.get(timeout=1)
            # Transcribe short chunks
            segments, _ = model.transcribe(audio_chunk, beam_size=1)
            text = " ".join([s.text for s in segments]).strip()
            if text:
                full_text += " " + text
                transcript_box.text_area("Transcript", full_text.strip(), height=300)
        except queue.Empty:
            continue
