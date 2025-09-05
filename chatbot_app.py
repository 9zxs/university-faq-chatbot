import streamlit as st
import joblib, random, json, re
from pathlib import Path
from deep_translator import GoogleTranslator
from langdetect import detect
import datetime
import plotly.express as px
import pandas as pd

# =============================
# Configuration & Paths
# =============================
DATA_PATH = Path(__file__).resolve().parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
LOG_PATH = Path(__file__).resolve().parent / "data" / "chat_logs.json"
KEYWORDS_PATH = Path(__file__).resolve().parent / "data" / "lang_keywords.json"
FEEDBACK_PATH = Path(__file__).resolve().parent / "data" / "feedback.json"

# =============================
# Logging
# =============================
def log_interaction(user_text, detected_lang, translated_input, predicted_tag, bot_reply, confidence=0.0, feedback=None):
    try:
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "user_text": user_text,
            "detected_lang": detected_lang,
            "translated_input": translated_input,
            "predicted_tag": predicted_tag,
            "bot_reply": bot_reply,
            "confidence": confidence,
            "session_id": st.session_state.get("session_id", "unknown"),
            "feedback": feedback
        })

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"⚠️ Could not save log: {e}")

# =============================
# Load model and responses
# =============================
@st.cache_resource
def load_model_and_data():
    clf = joblib.load(MODEL_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    responses = {intent.get("tag") or intent.get("intent"): intent.get("responses", []) for intent in data["intents"]}
    return clf, responses

clf, responses = load_model_and_data()

# =============================
# Session initialization
# =============================
def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []

init_session()

# =============================
# Language detection
# =============================
def detect_supported_lang(text):
    if KEYWORDS_PATH.exists():
        with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
            lang_keywords = json.load(f)
    else:
        lang_keywords = {"ms": [], "zh-CN": []}

    t = text.lower()
    ms_score = sum(1 for word in lang_keywords.get("ms", []) if word in t)
    zh_score = sum(1 for word in text if word in lang_keywords.get("zh-CN", []))

    if ms_score > 0:
        return "ms", ms_score / len(text.split())
    if zh_score > 0:
        return "zh-CN", zh_score / len(text)

    try:
        detected = detect(text)
        confidence = 0.8
        if detected in ["en"]: return "en", confidence
        elif detected in ["ms", "id"]: return "ms", confidence
        elif detected in ["zh", "zh-cn", "zh-tw"]: return "zh-CN", confidence
        else: return "en", 0.5
    except:
        return "en", 0.3

# =============================
# Context-aware response
# =============================
def get_contextual_response(tag, user_text, conversation_history):
    base_responses = responses.get(tag, responses.get("fallback", ["Sorry, I didn't understand that."]))
    if tag == "greeting" and len(conversation_history) > 2:
        base_responses = ["Welcome back! How can I help you today?", "Hello again! What would you like to know?"]
    follow_ups = {
        "admissions_requirements": "\n\n💡 You might also want to ask about tuition fees or scholarship opportunities.",
        "tuition_fees": "\n\n💡 Don't forget to check our scholarship programs!",
        "scholarship": "\n\n💡 Would you like to know about the application deadlines?",
        "exam_schedule": "\n\n💡 Need help with library hours for studying?",
    }
    response = random.choice(base_responses)
    if tag in follow_ups:
        response += follow_ups[tag]
    return response

# =============================
# Bot reply (immediate)
# =============================
def bot_reply(user_text):
    detected_lang, lang_confidence = detect_supported_lang(user_text)
    translated_input = user_text
    if detected_lang != "en":
        try:
            translated_input = GoogleTranslator(source="auto", target="en").translate(user_text)
        except:
            translated_input = user_text
    try:
        tag = clf.predict([translated_input.lower()])[0]
        confidence = 0.7
    except:
        tag = "fallback"
        confidence = 0.1

    reply_en = get_contextual_response(tag, user_text, st.session_state.conversation_context)
    reply = reply_en
    if detected_lang != "en":
        try:
            reply = GoogleTranslator(source="en", target=detected_lang).translate(reply_en)
        except:
            reply = reply_en

    st.session_state.conversation_context.append({
        "user": user_text,
        "bot": reply,
        "intent": tag,
        "confidence": confidence
    })
    if len(st.session_state.conversation_context) > 5:
        st.session_state.conversation_context.pop(0)

    st.session_state.history.append(("You", user_text))
    st.session_state.history.append(("Bot", reply))
    log_interaction(user_text, detected_lang, translated_input, tag, reply, confidence)
    # Clear input after submit
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
        st.success("Thank you for your feedback! 🙏")
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
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Conversations", len(df))
        with col2: st.metric("Unique Sessions", df['session_id'].nunique() if 'session_id' in df else "N/A")
        with col3: st.metric("Avg Confidence", f"{df['confidence'].mean():.2f}" if 'confidence' in df else "0")
        with col4: st.metric("Languages Used", df['detected_lang'].nunique() if 'detected_lang' in df else "N/A")
        col1, col2 = st.columns(2)
        with col1:
            if 'predicted_tag' in df:
                intent_counts = df['predicted_tag'].value_counts()
                fig = px.bar(x=intent_counts.index, y=intent_counts.values, title="Most Common Questions")
                fig.update_layout(xaxis_title="Intent", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if 'detected_lang' in df:
                lang_counts = df['detected_lang'].value_counts()
                fig = px.pie(values=lang_counts.values, names=lang_counts.index, title="Language Usage")
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("Recent Conversations")
        recent_df = df.tail(10)[['timestamp', 'user_text', 'predicted_tag', 'confidence']].copy()
        if 'confidence' in recent_df:
            recent_df['confidence'] = recent_df['confidence'].round(2)
        st.dataframe(recent_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# =============================
# App UI
# =============================
st.set_page_config(page_title="🎓 University FAQ Chatbot", page_icon="🤖", layout="wide")
logo_path = Path(__file__).resolve().parent / "data" / "university_logo.png"
col1, col2 = st.columns([1,4])
with col1:
    if logo_path.exists():
        st.image(str(logo_path), width=80)
with col2:
    st.title("🎓 University FAQ Chatbot")
    st.caption("Multilingual support: English • Malay • 中文")

# Sidebar
with st.sidebar:
    st.subheader("ℹ️ Info")
    st.info("This AI chatbot helps answer questions about:\n• Admissions\n• Tuition & Scholarships\n• Exams\n• Library\n• Housing\n• Office Hours")
    st.subheader("Session")
    st.text(f"Session ID: {st.session_state.session_id}")
    st.text(f"Messages: {len(st.session_state.history)}")
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []
        st.session_state.conversation_context = []

# =============================
# Theme-aware chat CSS
# =============================
st.markdown("""
<style>
.user-bubble, .bot-bubble {
    padding: 10px;
    border-radius: 10px;
    margin: 5px;
    max-width: 70%;
    word-wrap: break-word;
}

/* Light theme */
@media (prefers-color-scheme: light) {
    .user-bubble { background-color: #DCF8C6; color: #000; align-self: flex-end; }
    .bot-bubble { background-color: #F1F0F0; color: #000; align-self: flex-start; }
}

/* Dark theme */
@media (prefers-color-scheme: dark) {
    .user-bubble { background-color: #4A8B4E; color: #fff; align-self: flex-end; }
    .bot-bubble { background-color: #3A3A3A; color: #fff; align-self: flex-start; }
}

.chat-container {
    display: flex;
    flex-direction: column;
}
</style>
""", unsafe_allow_html=True)

# =============================
# Chat and Analytics Tabs
# =============================
tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])
with tab1:
    # Chat display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for speaker, msg in st.session_state.history:
        bubble_class = "user-bubble" if speaker == "You" else "bot-bubble"
        st.markdown(f'<div class="{bubble_class}">{speaker}: {msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input box with immediate response
    st.text_input("Ask me anything...", key="input", on_change=lambda: bot_reply(st.session_state.input))

with tab2:
    show_analytics()
