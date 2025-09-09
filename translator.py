import os
import streamlit as st
import tempfile
from gtts import gTTS
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
st.set_page_config(page_title="🌍 Gemini Translator", page_icon="🌐")
st.title("🌍 Language Translator with Voice Output")

input_text = st.text_area("✏️ Enter text to translate", height=120)

target_lang = st.selectbox(
    "🌐 Select target language",
    list(LANG_CODE_MAP.keys())
)

if st.button("Translate & Speak"):
    if input_text.strip():
        with st.spinner("Translating..."):
            try:
                # Get the language code for gTTS
                gtts_lang_code = LANG_CODE_MAP.get(target_lang, "en") 
                
                # Perform translation
                translated_output = translate_text(input_text, target_lang)
                
                st.markdown(f"### 📄 Translated Text ({target_lang}):")
                st.success(translated_output)
                
                # Convert translated text to speech and provide audio player
                audio_file_path = text_to_speech(translated_output, gtts_lang_code)
                if audio_file_path:
                    st.audio(audio_file_path, format="audio/mp3")
                    # Clean up the temporary file after use
                    os.remove(audio_file_path)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter text to translate.")
