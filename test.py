from gtts import gTTS
from pydub import AudioSegment
import streamlit as st
import tempfile

text = "This is a sample text to be converted to speech."

# Create a gTTS object
tts = gTTS(text)
audio_file="hlo.mp3"
# Save the speech as a temporary file
#audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
tts.save(audio_file)

# Load the audio file using pydub
audio = AudioSegment.from_mp3(audio_file)

# Play the audio file on Streamlit
st.audio(audio_file)

# Close and delete the temporary file
audio_file.close()
