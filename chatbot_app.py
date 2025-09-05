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
# Enhanced logging with sentiment tracking
# =============================
def log_interaction(user_text, detected_lang, translated_input, predicted_tag, bot_reply, confidence=0.0, feedback=None):
    try:
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_text": user_text,
            "detected_lang": detected_lang,
            "translated_input": translated_input,
            "predicted_tag": predicted_tag,
            "bot_reply": bot_reply,
            "confidence": confidence,
            "session_id": st.session_state.get("session_id", "unknown"),
            "feedback": feedback
        }
        
        logs.append(log_entry)

        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.warning(f"⚠️ Could not save log: {e}")

# =============================
# Enhanced model loading
# =============================
@st.cache_resource
def load_model_and_data():
    clf = joblib.load(MODEL_PATH)
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    responses = {}
    for intent in data["intents"]:
        tag = intent.get("tag") or intent.get("intent")
        responses[tag] = intent.get("responses", [])
    
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
    
    if "user_satisfaction" not in st.session_state:
        st.session_state.user_satisfaction = []
    
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []

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
        if detected in ["en"]:
            return "en", confidence
        elif detected in ["ms", "id"]:
            return "ms", confidence
        elif detected in ["zh", "zh-cn", "zh-tw"]:
            return "zh-CN", confidence
        else:
            return "en", 0.5
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
# Bot reply with confidence
# =============================
def bot_reply(user_text):
    detected_lang, lang_confidence = detect_supported_lang(user_text)
    
    if detected_lang != "en":
        try:
            translated_input = GoogleTranslator(source="auto", target="en").translate(user_text)
        except:
            translated_input = user_text
    else:
        translated_input = user_text

    try:
        tag = clf.predict([translated_input.lower()])[0]
        if hasattr(clf, "decision_function"):
            confidence_scores = clf.decision_function([translated_input.lower()])[0]
            confidence = max(confidence_scores) if len(confidence_scores) > 1 else confidence_scores[0]
            confidence = 1 / (1 + abs(confidence))
        else:
            confidence = 0.7
    except Exception:
        tag = "fallback"
        confidence = 0.1

    reply_en = get_contextual_response(tag, user_text, st.session_state.conversation_context)

    if detected_lang != "en":
        try:
            reply = GoogleTranslator(source="en", target=detected_lang).translate(reply_en)
        except:
            reply = reply_en
    else:
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
    return confidence

# =============================
# Feedback
# =============================
def save_feedback(rating, comment=""):
    try:
        if FEEDBACK_PATH.exists():
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        else:
            feedback_data = []
        
        feedback_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": st.session_state.session_id,
            "rating": rating,
            "comment": comment,
            "conversation_length": len(st.session_state.history)
        }
        
        feedback_data.append(feedback_entry)
        
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2)
        
        st.success("Thank you for your feedback! 🙏")
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# =============================
# Analytics Dashboard
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
# Main App Configuration
# =============================
st.set_page_config(page_title="🎓 University FAQ Chatbot", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
init_session()

# Header
logo_path = Path(__file__).resolve().parent / "data" / "university_logo.png"
col1, col2 = st.columns([1, 4])
with col1:
    if logo_path.exists():
        st.image(str(logo_path), width=80)
with col2:
    st.title("🎓 University FAQ Chatbot")
    st.caption("Multilingual support: English • Malay • 中文")

# Sidebar
with st.sidebar:
    with st.expander("ℹ️ Info", expanded=True):
        st.info(
            "This AI chatbot helps answer questions about:\n\n"
            "• 📚 Admissions & Requirements\n"
            "• 💰 Tuition Fees & Scholarships\n"
            "• 📅 Exam Schedules\n"
            "• 📖 Library Services\n"
            "• 🏠 Housing & Hostels\n"
            "• ⏰ Office Hours"
        )
    with st.expander("🔧 Session Info", expanded=True):
        st.text(f"Session ID: {st.session_state.session_id}")
        st.text(f"Messages: {len(st.session_state.history)}")
        if st.button("🗑️ Clear Chat"):
            st.session_state.history = []
            st.session_state.conversation_context = []
            st.rerun()

    if len(st.session_state.history) > 0:
        st.subheader("📝 Feedback")
        rating = st.select_slider(
            "Rating",
            options=[1,2,3,4,5],
            format_func=lambda x: "⭐" * x,
            key=f"rating_{len(st.session_state.history)}"
        )
        comment = st.text_area("Suggestions (optional)", placeholder="Any improvements?", key=f"comment_{len(st.session_state.history)}")
        if st.button("Submit Feedback"):
            save_feedback(rating, comment)

# Tabs
tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])
with tab1:
    st.subheader("🔍 Quick Questions")
    col1, col2, col3, col4 = st.columns(4)
    quick_buttons = [
        ("📚 Admission", "what are the admission requirements"),
        ("💰 Tuition", "how much is the tuition fee"),
        ("📅 Exam Dates", "when are the exams"),
        ("🏠 Housing", "how can I apply for hostels")
    ]
    for col, (label, query) in zip([col1, col2, col3, col4], quick_buttons):
        if col.button(label, use_container_width=True):
            bot_reply(query)
    
    if user_input := st.chat_input("Ask me anything about the university..."):
        bot_reply(user_input)

    st.markdown("""
    <style>
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            display: flex;
        }

        .chat_input{
            max-height: 400px;
            overflow-y: auto;
            display: flex;
            flex-direction: column-reverse; /* this puts new messages at the bottom, above input */
        }
        
        /* User messages (right side) */
        .chat-user {
            background-color: #DCF8C6;
            float: right;
            clear: both;
            text-align: right;
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
        
        /* Bot messages (left side) */
        .chat-bot {
            background-color: #F1F0F0;
            float: left;
            clear: both;
            text-align: left;
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
        
        /* Dark mode adjustments */
        @media (prefers-color-scheme: dark) {
            .chat-bot { background-color: #2E2E2E; }
            .chat-user { background-color: #3A523A; }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for speaker, msg in st.session_state.history:
        if speaker == "You":
            st.markdown(f'<div class="chat-user">You: {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">Bot: {msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    show_analytics()
