import streamlit as st
import json, random, datetime, uuid
from pathlib import Path
from deep_translator import GoogleTranslator
from langdetect import detect
import pandas as pd
from difflib import SequenceMatcher
import plotly.express as px

# =============================
# File paths
# =============================
BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "data" / "university_faq_dataset.csv"
LOG_PATH = BASE_DIR / "data" / "chat_logs.json"
FEEDBACK_PATH = BASE_DIR / "data" / "feedback.json"

# =============================
# Supported languages
# =============================
SUPPORTED_LANGS = {
    "en": "en",
    "zh-cn": "zh-CN"
}

# =============================
# Load knowledge base
# =============================
@st.cache_resource
def load_knowledge_base():
    df = pd.read_csv(KB_PATH)
    df["question"] = df["question"].str.strip().str.lower()
    return df

knowledge_base = load_knowledge_base()

# =============================
# Utils
# =============================
def translate_text(text, target_lang):
    try:
        if target_lang == "en":
            return text
        if target_lang == "zh-CN":
            return GoogleTranslator(source="en", target="zh-CN").translate(text)
        return text
    except:
        return text

def log_interaction(user_text, detected_lang, translated_input, bot_reply, confidence):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": st.session_state.session_id,
        "user_text": user_text,
        "detected_lang": detected_lang,
        "translated_input": translated_input,
        "bot_reply": bot_reply,
        "confidence": confidence
    }
    logs = []
    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
    logs.append(log_entry)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

# =============================
# Fuzzy matching
# =============================
def find_best_matches(user_input, questions, threshold=0.4):
    user_input_lower = user_input.lower()
    matches = []
    for q in questions:
        ratio = SequenceMatcher(None, user_input_lower, q).ratio()
        if ratio >= threshold:
            matches.append((q, ratio))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches

def get_csv_response(user_input, detected_lang="en"):
    fallback_response = {
        "en": "Sorry, I don’t know that yet. Please contact the admin office.",
        "zh-CN": "抱歉，我还不知道。请联系管理办公室。"
    }

    try:
        # Step 1: Translate user input to English if it's Chinese
        translated_input = user_input
        if detected_lang == "zh-CN":
            translated_input = GoogleTranslator(source="zh-CN", target="en").translate(user_input)

        # Step 2: Fuzzy match in English CSV
        questions = knowledge_base["question"].tolist()
        matches = find_best_matches(translated_input, questions, threshold=0.4)

        if matches:
            matched_q = matches[0][0]
            answer = knowledge_base.loc[knowledge_base["question"].str.lower() == matched_q.lower(), "answer"].values[0]
            response = answer
            confidence = matches[0][1]
        else:
            response = fallback_response[detected_lang]
            confidence = 0.0

        # Step 3: Translate back to Chinese if needed
        if detected_lang == "zh-CN" and response != fallback_response["zh-CN"]:
            response = GoogleTranslator(source="en", target="zh-CN").translate(response)

        return response, confidence, translated_input

    except Exception as e:
        print("Error in get_csv_response:", e)
        return fallback_response[detected_lang], 0.0, user_input

# =============================
# Session state
# =============================
if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# =============================
# Bot reply
# =============================
def bot_reply(user_text):
    detected_lang = "en"
    try:
        lang = detect(user_text)
        detected_lang = SUPPORTED_LANGS.get(lang.lower(), "en")
    except:
        detected_lang = "en"

    # Greetings
    greetings_en = ["hi", "hello", "hey"]
    greetings_zh = ["你好", "嗨"]
    if user_text.lower() in greetings_en or user_text in greetings_zh:
        reply = "Hello! How can I help you?" if detected_lang=="en" else "您好！我能帮您什么吗？"
        confidence = 1.0
    # Explicit time query (exact match)
    elif user_text.lower() in ["time", "what time is it"] or user_text in ["时间", "现在几点"]:
        reply = f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}." if detected_lang=="en" else f"当前时间是 {datetime.datetime.now().strftime('%H:%M:%S')}。"
        confidence = 1.0
    else:
        reply, confidence, _ = get_csv_response(user_text, detected_lang)

    # Save conversation
    st.session_state.history.append(("You", user_text))
    st.session_state.history.append(("Bot", reply))
    log_interaction(user_text, detected_lang, user_text, reply, confidence)
    st.session_state.input = ""

# =============================
# Feedback
# =============================
def save_feedback(rating, comment=""):
    try:
        feedback_data = []
        if FEEDBACK_PATH.exists():
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        feedback_data.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": st.session_state.session_id,
            "rating": rating,
            "comment": comment,
            "conversation_length": len(st.session_state.history)
        })
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2)
        st.success("Thank you for your feedback! 🙏" if "en" else "感谢您的反馈！🙏")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# =============================
# Analytics
# =============================
def show_analytics():
    st.header("📊 Analytics Dashboard")
    if not LOG_PATH.exists():
        st.warning("No conversation data available yet.")
        return
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
        if not logs:
            st.warning("No conversation data available yet.")
            return
        df = pd.DataFrame(logs)

        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Conversations", len(df))
        with col2: st.metric("Unique Sessions", df['session_id'].nunique() if 'session_id' in df else "N/A")
        with col3: st.metric("Languages Used", df['detected_lang'].nunique() if 'detected_lang' in df else "N/A")

        col1, col2 = st.columns(2)
        with col1:
            if 'bot_reply' in df:
                reply_counts = df['bot_reply'].value_counts()
                fig = px.bar(x=reply_counts.index, y=reply_counts.values, title="Most Common Answers")
                fig.update_layout(xaxis_title="Answer", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'detected_lang' in df:
                lang_counts = df['detected_lang'].value_counts()
                fig = px.pie(values=lang_counts.values, names=lang_counts.index, title="Language Usage")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Conversations")
        recent_df = df.tail(10)[['timestamp', 'user_text', 'bot_reply', 'confidence']].copy()
        if 'confidence' in recent_df:
            recent_df['confidence'] = recent_df['confidence'].round(2)
        st.dataframe(recent_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# =============================
# App UI
# =============================
st.set_page_config(page_title="🎓 University FAQ Chatbot", page_icon="🤖", layout="wide")
logo_path = BASE_DIR / "data" / "university_logo.png"
col1, col2 = st.columns([1,4])
with col1:
    if logo_path.exists():
        st.image(str(logo_path), width=80)
with col2:
    st.title("🎓 University FAQ Chatbot")
    st.caption("Multilingual support: English • 中文")

with st.sidebar:
    st.subheader("ℹ️ Info")
    st.info("This AI chatbot helps answer questions about:\n• Admissions\n• Tuition & Scholarships\n• Exams\n• Library\n• Housing\n• Office Hours")
    st.subheader("Session")
    st.text(f"Session ID: {st.session_state.session_id}")
    st.text(f"Messages: {len(st.session_state.history)}")
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []

st.markdown("""
<style>
.chat-container { display: flex; flex-direction: column; height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 10px;}
.user-bubble, .bot-bubble { padding: 10px; border-radius: 10px; margin: 5px; max-width: 70%; word-wrap: break-word; }
@media (prefers-color-scheme: light) { .user-bubble { background-color: #DCF8C6; color: #000; align-self: flex-end; } .bot-bubble { background-color: #F1F0F0; color: #000; align-self: flex-start; } }
@media (prefers-color-scheme: dark) { .user-bubble { background-color: #4A8B4E; color: #fff; align-self: flex-end; } .bot-bubble { background-color: #3A3A3A; color: #fff; align-self: flex-start; } }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])
with tab1:
    chat_html = '<div class="chat-container" id="chat-box">'
    for speaker, msg in st.session_state.history:
        bubble_class = "user-bubble" if speaker=="You" else "bot-bubble"
        chat_html += f'<div class="{bubble_class}">{speaker}: {msg}</div>'
    chat_html += '</div>'
    chat_html += "<script>var chatBox=document.getElementById('chat-box');if(chatBox){chatBox.scrollTop=chatBox.scrollHeight;}</script>"
    st.markdown(chat_html, unsafe_allow_html=True)
    st.text_input("Ask me anything...", key="input", on_change=lambda: bot_reply(st.session_state.input))

with tab2:
    show_analytics()
