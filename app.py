import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import model_predict
import av
import tempfile
from pydub import AudioSegment
import os
import time

st.set_page_config(page_title="SpeakSafe", layout="wide")

st.markdown("""
    <style>
        .title-text {
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(to right, purple, violet);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: fadeIn 2s ease-in-out;
        }
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem;
            text-align: center;
            font-weight: bold;
            background-color: #f0f2f6;
            color: #6c757d;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="SpeakSafe Menu",
        options=["Home", "Upload Audio", "Record Audio"],
        icons=["house", "cloud-upload", "mic"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "icon": {"color": "purple", "font-size": "25px"},
            "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#e0bbff"},
        }
    )

if selected == "Home":
    st.markdown("<div class='title-text'>🛡️ Welcome to SpeakSafe</div>", unsafe_allow_html=True)
    st.write("""
    SpeakSafe is your real-time guardian against deepfake audio threats. 
    Using powerful machine learning algorithms, this tool detects whether an audio sample is real or AI-generated.
    
    🔍 Features:
    - Upload an audio file and detect if it's real or fake.
    - Record your voice directly and test its authenticity.
    - Get instant results with confidence levels.
    
    🚀 Built for journalists, podcasters, researchers, and anyone concerned about voice-based misinformation.
    
    💡 Try it out from the left menu!
    """)

elif selected == "Upload Audio":
    st.markdown("<div class='title-text'>📤 Upload Audio File</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
        temp_file.close()

        if uploaded_file.type == "audio/mpeg":
            sound = AudioSegment.from_mp3(temp_path)
            temp_path = temp_path.replace(".tmp", ".wav")
            sound.export(temp_path, format="wav")

        st.audio(temp_path, format='audio/wav')

        result, confidence = model_predict(temp_path)
        if result == 1:
            st.success(f"✅ Real Audio. Confidence: {confidence:.2f}")
        else:
            st.error(f"❌ Fake Audio Detected! Confidence: {confidence:.2f}")

elif selected == "Record Audio":
    st.markdown("<div class='title-text'>🎤 Record Your Audio</div>", unsafe_allow_html=True)

    class AudioProcessor:
        def __init__(self):
            self.audio = b""

        def recv(self, frame):
            self.audio += frame.to_ndarray().tobytes()
            return av.AudioFrame.from_ndarray(frame.to_ndarray(), layout="mono")

    ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDRECV,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True
    )

    if ctx.audio_receiver:
        audio_bytes = b""
        with ctx.audio_receiver:
            st.info("Recording... Press stop when done.")
            while ctx.state.playing:
                try:
                    audio_frame = ctx.audio_receiver.recv()
                    audio_bytes += audio_frame.to_ndarray().tobytes()
                except Exception as e:
                    print(e)
                    break

        if audio_bytes:
            wav_path = os.path.join("temp_recorded.wav")
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)

            st.audio(wav_path, format='audio/wav')
            result, confidence = model_predict(wav_path)
            if result == 1:
                st.success(f"✅ Real Audio. Confidence: {confidence:.2f}")
            else:
                st.error(f"❌ Fake Audio Detected! Confidence: {confidence:.2f}")


st.markdown("""
    <div class="footer">✨ Made by Bhavana using Streamlit and Machine Learning 💜</div>
""", unsafe_allow_html=True)

