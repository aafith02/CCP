import os
import streamlit as st
import tempfile
from gtts import gTTS
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()
GOOGLE_API_KEY =""

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# Translator function
def translate_text(text: str, target_language: str) -> str:
    prompt = f"""Translate the following text to {target_language}:\n\n{text}"""
    chain = RunnableLambda(lambda input: llm.invoke([HumanMessage(content=input)]))
    result = chain.invoke(prompt)
    return result.content.strip()

# Text-to-speech function
def text_to_speech(text: str, lang_code: str):
    """Converts text to speech using gTTS and returns the path to the audio file."""
    if not text:
        return None
    try:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmpfile:
            tts = gTTS(text, lang=lang_code)
            tts.save(tmpfile.name)
            return tmpfile.name
    except Exception as e:
        st.error(f"❌ TTS Error: {str(e)}")
        return None

# Speech-to-text function
def speech_to_text() -> str:
    """Converts speech from microphone input to text."""
    r = sr.Recognizer()
    with st.spinner("🎧 Listening... Please speak now."):
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            
            # Using Google Web Speech API for transcription
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("❌ Sorry, I could not understand the audio. Please try again.")
            return ""
        except sr.RequestError as e:
            st.error(f"❌ Could not request results from Google Speech Recognition service; {e}")
            return ""
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
            return ""

# Map for gTTS language codes
LANG_CODE_MAP = {
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-cn",
    "Arabic": "ar",
    "Urdu": "ur",
    "Japanese": "ja",
    "Russian": "ru",
    "Hindi": "hi",
    "Italian": "it"
}

# Streamlit UI
st.set_page_config(page_title="🌍 Gemini Speech-to-Speech Translator", page_icon="🌐")
st.title("🗣️ Speech-to-Speech Translator")
st.write("Click the 'Speak & Translate' button and say something into your microphone. The app will transcribe, translate, and speak the result.")

target_lang = st.selectbox(
    "🌐 Select target language",
    list(LANG_CODE_MAP.keys())
)

if st.button("🎙️ Speak & Translate"):
    # Step 1: Get speech input from the user and convert to text
    user_speech_text = speech_to_text()
    
    if user_speech_text:
        st.info(f"🎤 You said: **{user_speech_text}**")
        
        # Step 2: Translate the transcribed text
        with st.spinner("Translating..."):
            try:
                # Get the language code for gTTS
                gtts_lang_code = LANG_CODE_MAP.get(target_lang, "en") 
                
                # Perform translation
                translated_output = translate_text(user_speech_text, target_lang)
                
                st.markdown(f"### 📄 Translated Text ({target_lang}):")
                st.success(translated_output)
                
                # Step 3: Convert translated text to speech and provide audio player
                audio_file_path = text_to_speech(translated_output, gtts_lang_code)
                if audio_file_path:
                    st.audio(audio_file_path, format="audio/mp3")
                    # Clean up the temporary file after use
                    os.remove(audio_file_path)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
