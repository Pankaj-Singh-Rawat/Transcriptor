import streamlit as st
import openai
import tempfile
import sounddevice as sd
import wavio

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Better Transcriptor 🎙️")
st.write("Click record, speak your answer, and get instant transcription.")

duration = st.number_input("Recording duration (seconds)", min_value=3, max_value=60, value=5)

if st.button("🎤 Record"):
    st.write("Recording...")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wavio.write(tmp.name, audio, fs, sampwidth=2)
        audio_file = open(tmp.name, "rb")
    
    # Send to OpenAI Whisper
    transcript = openai.audio.transcriptions.create(
        model="gpt-4o-transcribe", 
        file=audio_file
    )
    
    st.success("✅ Transcription:")
    st.write(transcript["text"])
