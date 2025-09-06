import streamlit as st
import json, random, datetime, uuid, re, os
from pathlib import Path
from deep_translator import GoogleTranslator
from langdetect import detect
import pandas as pd
from difflib import SequenceMatcher
import plotly.express as px

# =============================
# Configuration and Constants
# =============================
BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "data" / "university_faq_dataset.csv"
LOG_PATH = BASE_DIR / "data" / "chat_logs.json"
FEEDBACK_PATH = BASE_DIR / "data" / "feedback.json"

# Ensure data directory exists
(BASE_DIR / "data").mkdir(exist_ok=True)

# Supported languages (only English and Chinese)
SUPPORTED_LANGS = {
    "en": "English",
    "zh-cn": "中文"
}

LANG_CODES = {
    "en": "en",
    "zh-cn": "zh-CN"
}

# =============================
# Error Handling and Validation
# =============================
class ChatbotException(Exception):
    pass

def validate_input(text):
    if not text or not isinstance(text, str):
        raise ChatbotException("Invalid input provided")
    sanitized = re.sub(r'[<>"\']', '', text.strip())
    if len(sanitized) > 500:
        raise ChatbotException("Input too long. Limit to 500 characters.")
    return sanitized

def safe_translate(text, target_lang, source_lang="auto"):
    try:
        if not text or target_lang == "en":
            return text
        translator = GoogleTranslator(source=source_lang, target=LANG_CODES.get(target_lang, "en"))
        result = translator.translate(text)
        return result if result else text
    except Exception as e:
        st.warning(f"Translation service unavailable: {str(e)}")
        return text

def safe_detect_language(text):
    try:
        if not text:
            return "en"
        detected = detect(text)
        return LANG_CODES.get(detected.lower(), "en")
    except Exception:
        return "en"

# =============================
# Load Knowledge Base
# =============================
@st.cache_resource(ttl=3600)
def load_knowledge_base():
    try:
        if not KB_PATH.exists():
            sample_data = {
                "question": [
                    "what are the admission requirements",
                    "how much is tuition fee",
                    "when do classes start",
                    "what courses are available",
                    "how to apply for scholarship"
                ],
                "answer": [
                    "Admission requirements include a high school diploma, English proficiency test, and completed application form.",
                    "Tuition fees vary by program. Domestic: RM10,000 per semester. International: RM15,000 per semester.",
                    "Classes typically start in January, May, and September.",
                    "Programs: Computer Science, IT, Business, Engineering, Nursing.",
                    "Scholarships available based on merit, need, or talent. Apply through our portal."
                ]
            }
            df = pd.DataFrame(sample_data)
            KB_PATH.parent.mkdir(exist_ok=True)
            df.to_csv(KB_PATH, index=False)
        else:
            df = pd.read_csv(KB_PATH)
        df["question"] = df["question"].str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Error loading knowledge base: {str(e)}")
        return pd.DataFrame(columns=["question", "answer"])

# =============================
# Courses Data
# =============================
COURSES = {
    "computer science": {
        "description": "Computer Science prepares students to design and develop software systems.",
        "duration": "4 years (8 semesters)",
        "career_prospects": ["Software Developer", "Data Scientist", "AI Engineer"],
        "keywords": ["computer", "cs", "programming", "software", "coding"]
    },
    "information technology": {
        "description": "IT focuses on applying technology to solve business problems.",
        "duration": "3 years (6 semesters)",
        "career_prospects": ["IT Manager", "Network Administrator", "System Analyst"],
        "keywords": ["it", "information technology", "networking", "systems"]
    },
    "business": {
        "description": "Business studies prepare students for management roles.",
        "duration": "3 years (6 semesters)",
        "career_prospects": ["Business Manager", "Marketing Executive", "Financial Analyst"],
        "keywords": ["business", "management", "marketing", "finance", "mba"]
    }
}

# =============================
# Session Management
# =============================
def initialize_session():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "current_course" not in st.session_state:
        st.session_state.current_course = None
    if "user_language" not in st.session_state:
        st.session_state.user_language = "en"
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []

# =============================
# Logging
# =============================
def log_interaction(user_text, detected_lang, translated_input, bot_reply, confidence, intent=None):
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": st.session_state.session_id,
            "user_text": user_text,
            "detected_lang": detected_lang,
            "translated_input": translated_input,
            "bot_reply": bot_reply,
            "confidence": confidence,
            "intent": intent
        }
        logs = []
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append(log_entry)
        if len(logs) > 1000:
            logs = logs[-1000:]
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Logging error: {str(e)}")

# =============================
# Response Generation
# =============================
def get_enhanced_response(user_input, detected_lang="en"):
    try:
        user_input = validate_input(user_input)
        knowledge_base = load_knowledge_base()

        fallback_responses = {
            "en": "I'm sorry, I don't have information about that. Try asking about courses, admissions, fees, or facilities.",
            "zh-CN": "抱歉，我没有相关信息。请尝试询问课程、入学、费用或设施。"
        }

        translated_input = user_input
        if detected_lang != "en":
            translated_input = safe_translate(user_input, "en", detected_lang)
        user_input_lower = translated_input.lower()

        intent = classify_intent(user_input_lower)

        if intent == "greeting":
            response = handle_greeting(detected_lang)
            confidence = 1.0
        elif intent == "course_inquiry":
            response, confidence = handle_course_inquiry(user_input_lower, detected_lang)
        elif intent == "fees":
            response, confidence = handle_fees_inquiry(user_input_lower, detected_lang)
        elif intent == "time":
            response = handle_time_query(detected_lang)
            confidence = 1.0
        else:
            response, confidence = fuzzy_match_kb(translated_input, knowledge_base)
            if confidence < 0.3:
                response = fallback_responses.get(detected_lang, fallback_responses["en"])
                confidence = 0.0

        if detected_lang != "en" and confidence > 0:
            response = safe_translate(response, detected_lang)

        return response, confidence, translated_input, intent
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I encountered an error. Please try again.", 0.0, user_input, "error"

def classify_intent(user_input):
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
    course_keywords = ["course", "program", "study", "curriculum"]
    fee_keywords = ["fee", "cost", "tuition", "payment"]
    time_keywords = ["time", "when", "schedule", "calendar"]

    if any(word in user_input for word in greetings):
        return "greeting"
    elif any(word in user_input for word in course_keywords):
        return "course_inquiry"
    elif any(word in user_input for word in fee_keywords):
        return "fees"
    elif any(word in user_input for word in time_keywords):
        return "time"
    else:
        return "general"

def handle_greeting(lang):
    greetings = {
        "en": "Hello! 👋 Welcome to our University. How can I assist you today?",
        "zh-CN": "您好！👋 欢迎来到我们大学。我能为您做些什么？"
    }
    return greetings.get(lang, greetings["en"])

def handle_course_inquiry(user_input, lang):
    for course_name, course_data in COURSES.items():
        if any(keyword in user_input for keyword in course_data["keywords"]):
            response = f"📚 **{course_name.title()}**\n"
            response += f"**Description:** {course_data['description']}\n"
            response += f"**Duration:** {course_data['duration']}\n"
            response += f"**Career Prospects:** {', '.join(course_data['career_prospects'])}\n"
            response += "Would you like to know more about the curriculum or admission requirements?"
            st.session_state.current_course = course_name
            return response, 0.9

    response = "🎓 We offer the following programs:\n"
    for course_name in COURSES.keys():
        response += f"• **{course_name.title()}**\n"
    response += "Which program would you like to know more about?"
    return response, 0.8

def handle_fees_inquiry(user_input, lang):
    course = st.session_state.current_course
    if "international" in user_input:
        fee = "RM 15,000"
        student_type = "international"
    elif "domestic" in user_input or "local" in user_input:
        fee = "RM 10,000"
        student_type = "domestic"
    else:
        response = "💰 **Tuition Fees:**\n• Domestic: RM 10,000 per semester\n• International: RM 15,000 per semester\n"
        response += "Additional fees may apply. Want info about scholarships?"
        return response, 1.0

    if course:
        response = f"💰 Tuition fees for **{course.title()}**: **{fee}** per semester ({student_type} students)."
    else:
        response = f"💰 Tuition fees: **{fee}** per semester ({student_type} students)."
    return response, 1.0

def handle_time_query(lang):
    current_time = datetime.datetime.now()
    responses = {
        "en": f"🕒 Current time: {current_time.strftime('%I:%M %p')} on {current_time.strftime('%B %d, %Y')}",
        "zh-CN": f"🕒 当前时间：{current_time.strftime('%Y年%m月%d日 %H:%M')}"
    }
    return responses.get(lang, responses["en"])

def fuzzy_match_kb(user_input, knowledge_base):
    if knowledge_base.empty:
        return "Knowledge base is empty.", 0.0
    questions = knowledge_base["question"].tolist()
    matches = []
    user_words = set(user_input.lower().split())
    for idx, question in enumerate(questions):
        question_words = set(question.split())
        overlap_ratio = len(user_words.intersection(question_words)) / len(user_words.union(question_words)) if user_words.union(question_words) else 0
        seq_ratio = SequenceMatcher(None, user_input.lower(), question).ratio()
        combined_score = (overlap_ratio * 0.6) + (seq_ratio * 0.4)
        if combined_score > 0.3:
            matches.append((question, combined_score, idx))
    if matches:
        matches.sort(key=lambda x: x[1], reverse=True)
        best_match = matches[0]
        answer = knowledge_base.iloc[best_match[2]]["answer"]
        return answer, best_match[1]
    return "No relevant information found.", 0.0

# =============================
# Chat Interface
# =============================
def chat_interface():
    st.subheader("💬 Chat with our AI Assistant")
    display_chat()
    user_input = st.text_input("Type your message here...", key="chat_input")
    if user_input and not st.session_state.get("just_submitted", False):
        process_input(user_input)
        st.session_state.just_submitted = True
        st.experimental_rerun()
    else:
        st.session_state.just_submitted = False

def display_chat():
    if not st.session_state.history:
        st.info("👋 Welcome! Start a conversation by asking about programs, admissions, fees, or schedules.")
        return
    for speaker, message in st.session_state.history:
        if speaker == "You":
            st.markdown(f"**{speaker}:** {message}")
        else:
            st.markdown(f"**{speaker}:** {message}")

def process_input(user_input):
    detected_lang = safe_detect_language(user_input)
    response, confidence, translated_input, intent = get_enhanced_response(user_input, detected_lang)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))
    log_interaction(user_input, detected_lang, translated_input, response, confidence, intent)

# =============================
# Main
# =============================
def main():
    st.set_page_config(page_title="University FAQ Chatbot", layout="wide")
    initialize_session()
    st.title("🎓 University FAQ Chatbot")
    chat_interface()

if __name__ == "__main__":
    main()
