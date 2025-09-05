import streamlit as st
import joblib, random, json
from pathlib import Path
from gtts import gTTS
from deep_translator import GoogleTranslator
from langdetect import detect

# =============================
# Load model & responses
# =============================
DATA_PATH = Path(__file__).resolve().parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
clf = joblib.load(MODEL_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}

# =============================
# Page Configuration
# =============================
st.set_page_config(page_title="🎓 University FAQ Chatbot", page_icon="🤖", layout="wide")

logo_path = Path(__file__).resolve().parent / "data" / "university_logo.png"
if logo_path.exists():
    st.image(str(logo_path), width=120)
st.title("🎓 University FAQ Chatbot 🤖 (English / Malay / Chinese)")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This chatbot answers common questions about **university admissions, fees, exams, "
    "library, scholarships, and more.**\n\n"
    "💡 Powered by `scikit-learn`, `Streamlit`, `deep-translator`, and `langdetect`."
)

# =============================
# Custom CSS
# =============================
st.markdown("""
<style>
.chat-bubble {
    display: inline-block;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 5px;
    font-size: 16px;
    word-wrap: break-word;
    max-width: 70%;
    min-width: 50px;
    color: var(--text-color);
}
.user { background-color: #DCF8C6; float: right; clear: both; text-align: right; }
.bot { background-color: #F1F0F0; float: left; clear: both; text-align: left; }
@media (prefers-color-scheme: dark) {
    .bot { background-color: #2E2E2E; }
    .user { background-color: #3A523A; }
}
</style>
""", unsafe_allow_html=True)

# =============================
# Session state
# =============================
if "history" not in st.session_state:
    st.session_state.history = []

# =============================
# Language detection helper
# =============================
def detect_supported_lang(text):
    t = text.lower()

    # Malay keywords (force Malay detection for short texts)
    malay_keywords = ["apa", "khabar", "universiti", "yuran", "pelajar", "kuliah", "biasiswa"]
    if any(word in t for word in malay_keywords):
        return "ms"

    # Chinese keywords (force Chinese detection for short texts)
    chinese_keywords = ["你好", "大学", "学费", "考试", "奖学金"]
    if any(word in t for word in chinese_keywords):
        return "zh-CN"

    # Fallback to langdetect
    try:
        lang = detect(text)
    except:
        return "en"

    if lang in ["en"]:
        return "en"
    elif lang in ["ms", "id"]:   # Malay often detected as Indonesian
        return "ms"
    elif lang in ["zh", "zh-cn", "zh-tw"]:
        return "zh-CN"
    else:
        return "en"

def lang_label(lang_code):
    if lang_code == "en":
        return "🌍 Detected: English"
    elif lang_code == "ms":
        return "🌍 Detected: Malay"
    elif lang_code == "zh-CN":
        return "🌍 Detected: Chinese"
    return "🌍 Detected: English"

# =============================
# Bot reply (single clean version)
# =============================
def bot_reply(user_text):
    # Step 1: Detect language
    detected_lang = detect_supported_lang(user_text)

    # Step 2: Translate input → English if needed
    if detected_lang != "en":
        translated_input = GoogleTranslator(source="auto", target="en").translate(user_text)
    else:
        translated_input = user_text

    # Step 3: Predict intent
    try:
        tag = clf.predict([translated_input.lower()])[0]
    except Exception:
        tag = "fallback"

    # Step 4: Bot reply in English
    reply_en = random.choice(responses.get(tag, responses["fallback"]))

    # Step 5: Translate reply back
    if detected_lang != "en":
        reply = GoogleTranslator(source="en", target=detected_lang).translate(reply_en)
    else:
        reply = reply_en

    # Step 6: Save chat history
    st.session_state.history.append(("You", f"{user_text}\n\n_{lang_label(detected_lang)}_"))
    st.session_state.history.append(("Bot", reply))

# =============================
# Quick FAQ Buttons
# =============================
st.markdown("### 🔍 Quick Questions")
col1, col2, col3 = st.columns(3)

if col1.button("📚 Admission Requirements"):
    bot_reply("what are the admission requirements")

if col2.button("💰 Tuition Fees"):
    bot_reply("how much is the tuition fee")

if col3.button("📅 Exam Dates"):
    bot_reply("when are the exams")

# =============================
# Text Input
# =============================
if user_input := st.chat_input("Ask me anything about the university... (English, Malay, or Chinese)"):
    bot_reply(user_input)

# =============================
# Display chat history
# =============================
for speaker, msg in st.session_state.history:
    bubble_class = "user" if speaker == "You" else "bot"
    prefix = "🧑" if speaker == "You" else "🤖"
    st.markdown(
        f'<div class="chat-bubble {bubble_class}">{prefix} {msg}</div>',
        unsafe_allow_html=True
    )
