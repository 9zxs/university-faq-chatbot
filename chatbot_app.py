import streamlit as st
import json, random, datetime, uuid, re, os
from pathlib import Path
from deep_translator import GoogleTranslator
from langdetect import detect
import pandas as pd
from difflib import SequenceMatcher
import plotly.express as px
import plotly.graph_objects as go

# =============================
# Configuration and Constants
# =============================
BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "data" / "university_faq_dataset.csv"
LOG_PATH = BASE_DIR / "data" / "chat_logs.json"
FEEDBACK_PATH = BASE_DIR / "data" / "feedback.json"

# Ensure data directory exists
(BASE_DIR / "data").mkdir(exist_ok=True)

# Enhanced supported languages
SUPPORTED_LANGS = {
    "en": "English",
    "zh-cn": "中文",
    "ms": "Bahasa Malaysia",
    "ta": "தமிழ்"
}

LANG_CODES = {
    "en": "en",
    "zh-cn": "zh-CN",
    "ms": "ms",
    "ta": "ta"
}

# =============================
# Enhanced Error Handling and Validation
# =============================
class ChatbotException(Exception):
    """Custom exception for chatbot errors"""
    pass

def validate_input(text):
    """Validate and sanitize user input"""
    if not text or not isinstance(text, str):
        raise ChatbotException("Invalid input provided")
    
    # Remove potential harmful characters
    sanitized = re.sub(r'[<>"\']', '', text.strip())
    
    # Check length
    if len(sanitized) > 500:
        raise ChatbotException("Input too long. Please limit to 500 characters.")
    
    return sanitized

def safe_translate(text, target_lang, source_lang="auto"):
    """Safe translation with error handling"""
    try:
        if not text or target_lang == "en":
            return text
        
        translator = GoogleTranslator(source=source_lang, target=LANG_CODES.get(target_lang, "en"))
        result = translator.translate(text)
        return result if result else text
    except Exception as e:
        st.warning(f"Translation service temporarily unavailable: {str(e)}")
        return text

def safe_detect_language(text):
    """Safe language detection with fallback"""
    try:
        if not text:
            return "en"
        detected = detect(text)
        return LANG_CODES.get(detected.lower(), "en")
    except Exception:
        return "en"

# =============================
# Enhanced Data Loading with Caching
# =============================
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_knowledge_base():
    """Load knowledge base with error handling"""
    try:
        if not KB_PATH.exists():
            st.warning("Knowledge base file not found. Creating sample data...")
            # Create sample data if file doesn't exist
            sample_data = {
                "question": [
                    "what are the admission requirements",
                    "how much is tuition fee",
                    "when do classes start",
                    "what courses are available",
                    "how to apply for scholarship"
                ],
                "answer": [
                    "Admission requirements include a high school diploma with minimum GPA of 3.0, English proficiency test scores, and completed application form.",
                    "Tuition fees vary by program. Domestic students pay RM 10,000 per semester, international students pay RM 15,000 per semester.",
                    "Classes typically start in January, May, and September. Please check the academic calendar for exact dates.",
                    "We offer programs in Computer Science, Information Technology, Business, Engineering, and Nursing.",
                    "Scholarships are available based on academic merit, financial need, and special talents. Apply through our scholarship portal."
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
# Enhanced Course Data with More Details
# =============================
COURSES = {
    "computer science": {
        "description": "Computer Science prepares students to design, develop, and analyze software systems. Focuses on problem-solving, programming skills, and understanding computing theory.",
        "duration": "4 years (8 semesters)",
        "career_prospects": ["Software Developer", "Data Scientist", "AI Engineer", "System Analyst", "Cybersecurity Specialist"],
        "curriculum": {
            "Semester 1": ["Introduction to Programming", "Mathematics for Computing", "Computer Systems Fundamentals", "Communication Skills"],
            "Semester 2": ["Data Structures", "Object-Oriented Programming", "Discrete Mathematics", "Digital Logic Design"],
            "Semester 3": ["Algorithms", "Database Systems", "Web Development", "Operating Systems"],
            "Semester 4": ["Software Engineering", "Computer Networks", "Human-Computer Interaction", "Statistics"],
            "Semester 5": ["Artificial Intelligence", "Mobile Development", "Cybersecurity", "Elective 1"],
            "Semester 6": ["Machine Learning", "Cloud Computing", "Elective 2", "Project I"],
            "Semester 7": ["Advanced AI", "Data Analytics", "Elective 3", "Project II"],
            "Semester 8": ["Capstone Project", "Internship", "Industry Seminars", "Elective 4"]
        },
        "keywords": ["computer", "cs", "programming", "software", "coding"]
    },
    "information technology": {
        "description": "Information Technology focuses on applying technology to solve business problems. Emphasizes networking, system administration, and IT management.",
        "duration": "3 years (6 semesters)",
        "career_prospects": ["IT Manager", "Network Administrator", "System Analyst", "IT Consultant", "Database Administrator"],
        "curriculum": {
            "Semester 1": ["IT Fundamentals", "Mathematics for IT", "Computer Systems", "Communication Skills"],
            "Semester 2": ["Networking Basics", "Database Fundamentals", "Web Technologies", "Programming"],
            "Semester 3": ["Systems Analysis", "Software Development", "IT Security", "Operating Systems"],
            "Semester 4": ["Cloud Computing", "IT Project Management", "Data Analytics", "Elective 1"],
            "Semester 5": ["Network Administration", "IT Strategy", "Elective 2", "Internship"],
            "Semester 6": ["Capstone Project", "Industry Training", "Advanced Topics", "Portfolio"]
        },
        "keywords": ["it", "information technology", "networking", "systems"]
    },
    "business": {
        "description": "Business studies prepare students for management roles. Covers finance, marketing, operations, and strategic management.",
        "duration": "3 years (6 semesters)",
        "career_prospects": ["Business Manager", "Marketing Executive", "Financial Analyst", "HR Manager", "Entrepreneur"],
        "curriculum": {
            "Semester 1": ["Principles of Management", "Microeconomics", "Business Communication", "Accounting"],
            "Semester 2": ["Macroeconomics", "Marketing Fundamentals", "Business Law", "Statistics"],
            "Semester 3": ["Organizational Behavior", "Financial Management", "Operations Management", "Elective 1"],
            "Semester 4": ["Strategic Management", "Human Resources", "International Business", "Elective 2"],
            "Semester 5": ["Business Ethics", "Entrepreneurship", "Project Management", "Internship"],
            "Semester 6": ["Capstone Project", "Industry Seminar", "Portfolio", "Elective 3"]
        },
        "keywords": ["business", "management", "marketing", "finance", "mba"]
    }
}

# =============================
# Enhanced Session Management
# =============================
def initialize_session():
    """Initialize session state variables"""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "course_pending" not in st.session_state:
        st.session_state.course_pending = None
    if "current_course" not in st.session_state:
        st.session_state.current_course = None
    if "user_language" not in st.session_state:
        st.session_state.user_language = "en"
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []

# =============================
# Enhanced Logging System
# =============================
def log_interaction(user_text, detected_lang, translated_input, bot_reply, confidence, intent=None):
    """Enhanced logging with more metadata"""
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": st.session_state.session_id,
            "user_text": user_text,
            "detected_lang": detected_lang,
            "translated_input": translated_input,
            "bot_reply": bot_reply,
            "confidence": confidence,
            "intent": intent,
            "conversation_turn": len(st.session_state.history) // 2
        }
        
        logs = []
        if LOG_PATH.exists():
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        # Keep only last 1000 logs to prevent file bloat
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Logging error: {str(e)}")

# =============================
# Enhanced Response Generation
# =============================
def get_enhanced_response(user_input, detected_lang="en"):
    """Enhanced response generation with better context handling"""
    try:
        # Validate input
        user_input = validate_input(user_input)
        
        # Load knowledge base
        knowledge_base = load_knowledge_base()
        
        fallback_responses = {
            "en": "I'm sorry, I don't have information about that. Could you please rephrase your question or ask about our courses, admissions, fees, or facilities?",
            "zh-CN": "抱歉，我没有相关信息。您能重新表达您的问题或询问我们的课程、入学、费用或设施吗？",
            "ms": "Maaf, saya tidak mempunyai maklumat tentang itu. Bolehkah anda menyatakan semula soalan anda atau bertanya tentang kursus, kemasukan, yuran atau kemudahan kami?",
            "ta": "மன்னிக்கவும், என்னிடம் அதைப் பற்றிய தகவல் இல்லை. தயவுசெய்து உங்கள் கேள்வியை மறுபரிசீலனை செய்யுங்கள் அல்லது எங்கள் படிப்புகள், சேர்க்கை, கட்டணம் அல்லது வசதிகளைப் பற்றி கேட்கவும்?"
        }
        
        # Translate input if needed
        translated_input = user_input
        if detected_lang != "en":
            translated_input = safe_translate(user_input, "en", detected_lang)
        
        user_input_lower = translated_input.lower()
        
        # Intent classification
        intent = classify_intent(user_input_lower)
        
        # Handle different intents
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
            # Fuzzy match against knowledge base
            response, confidence = fuzzy_match_kb(translated_input, knowledge_base)
            if confidence < 0.3:
                response = fallback_responses.get(detected_lang, fallback_responses["en"])
                confidence = 0.0
        
        # Translate response if needed
        if detected_lang != "en" and confidence > 0:
            response = safe_translate(response, detected_lang)
        
        return response, confidence, translated_input, intent
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error. Please try again.", 0.0, user_input, "error"

def classify_intent(user_input):
    """Classify user intent"""
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
    course_keywords = ["course", "program", "study", "curriculum", "syllabus"]
    fee_keywords = ["fee", "cost", "price", "tuition", "payment"]
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
    """Handle greeting with personalized response"""
    greetings = {
        "en": f"Hello! 👋 Welcome to our University. I'm here to help you with information about our courses, admissions, fees, and more. How can I assist you today?",
        "zh-CN": "您好！👋 欢迎来到我们大学。我在这里为您提供有关课程、入学、费用等信息。今天我能为您做什么？",
        "ms": "Halo! 👋 Selamat datang ke Universiti kami. Saya di sini untuk membantu anda dengan maklumat tentang kursus, kemasukan, yuran dan banyak lagi. Bagaimana saya boleh membantu anda hari ini?",
        "ta": "வணக்கம்! 👋 எங்கள் பல்கலைக்கழகத்திற்கு வரவேற்கிறோம். படிப்புகள், சேர்க்கை, கட்டணம் மற்றும் பலவற்றைப் பற்றிய தகவல்களுடன் உங்களுக்கு உதவ நான் இங்கே இருக்கிறேன். இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?"
    }
    return greetings.get(lang, greetings["en"])

def handle_course_inquiry(user_input, lang):
    """Handle course-related inquiries"""
    # Check for specific course mentions
    for course_name, course_data in COURSES.items():
        if any(keyword in user_input for keyword in course_data["keywords"]):
            response = f"📚 **{course_name.title()}**\n\n"
            response += f"**Description:** {course_data['description']}\n\n"
            response += f"**Duration:** {course_data['duration']}\n\n"
            response += f"**Career Prospects:** {', '.join(course_data['career_prospects'])}\n\n"
            response += "Would you like to know more about the curriculum or admission requirements?"
            return response, 0.9
    
    # General course information
    response = "🎓 We offer the following programs:\n\n"
    for course_name in COURSES.keys():
        response += f"• **{course_name.title()}**\n"
    response += "\nWhich program would you like to know more about?"
    return response, 0.8

def handle_fees_inquiry(user_input, lang):
    """Handle fee-related inquiries"""
    course = st.session_state.current_course
    
    if "international" in user_input:
        student_type = "international"
        fee = "RM 15,000"
    elif "domestic" in user_input or "local" in user_input:
        student_type = "domestic"
        fee = "RM 10,000"
    else:
        response = "💰 **Tuition Fees:**\n\n"
        response += "• **Domestic Students:** RM 10,000 per semester\n"
        response += "• **International Students:** RM 15,000 per semester\n\n"
        response += "Additional fees may apply for laboratory, library, and other facilities.\n"
        response += "Would you like information about scholarships or payment plans?"
        return response, 1.0
    
    if course:
        response = f"💰 The tuition fees for **{course.title()}** are **{fee}** per semester for {student_type} students."
    else:
        response = f"💰 Tuition fees are **{fee}** per semester for {student_type} students."
    
    return response, 1.0

def handle_time_query(lang):
    """Handle time-related queries"""
    current_time = datetime.datetime.now()
    responses = {
        "en": f"🕒 Current time: {current_time.strftime('%I:%M %p')} on {current_time.strftime('%B %d, %Y')}",
        "zh-CN": f"🕒 当前时间：{current_time.strftime('%Y年%m月%d日 %H:%M')}",
        "ms": f"🕒 Masa sekarang: {current_time.strftime('%I:%M %p')} pada {current_time.strftime('%d %B %Y')}",
        "ta": f"🕒 தற்போதைய நேரம்: {current_time.strftime('%I:%M %p')} {current_time.strftime('%d %B %Y')}"
    }
    return responses.get(lang, responses["en"])

def fuzzy_match_kb(user_input, knowledge_base):
    """Enhanced fuzzy matching with better scoring"""
    if knowledge_base.empty:
        return "Knowledge base is empty.", 0.0
    
    questions = knowledge_base["question"].tolist()
    matches = []
    
    user_words = set(user_input.lower().split())
    
    for idx, question in enumerate(questions):
        # Calculate word overlap
        question_words = set(question.split())
        overlap = len(user_words.intersection(question_words))
        overlap_ratio = overlap / len(user_words.union(question_words)) if user_words.union(question_words) else 0
        
        # Calculate sequence similarity
        seq_ratio = SequenceMatcher(None, user_input.lower(), question).ratio()
        
        # Combined score
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
# Enhanced Analytics with Better Visualizations
# =============================
def show_enhanced_analytics():
    """Enhanced analytics dashboard"""
    st.header("📊 Enhanced Analytics Dashboard")
    
    # Load data
    logs = []
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception as e:
            st.error(f"Error loading logs: {e}")
    
    feedback_data = []
    if FEEDBACK_PATH.exists():
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        except Exception as e:
            st.error(f"Error loading feedback: {e}")
    
    # Summary metrics with better styling
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Conversations", len(logs), delta=None)
    with col2:
        unique_sessions = len(set([log.get("session_id", "") for log in logs]))
        st.metric("Unique Sessions", unique_sessions)
    with col3:
        languages = set([log.get("detected_lang", "en") for log in logs])
        st.metric("Languages", len(languages))
    with col4:
        st.metric("Feedback Entries", len(feedback_data))
    with col5:
        avg_confidence = sum([log.get("confidence", 0) for log in logs]) / len(logs) if logs else 0
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Detailed analytics
    if logs:
        df_logs = pd.DataFrame(logs)
        
        # Language distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Language Distribution")
            lang_counts = df_logs['detected_lang'].value_counts()
            fig = px.pie(values=lang_counts.values, names=lang_counts.index, 
                        title="User Language Preferences")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Confidence Score Distribution")
            fig = px.histogram(df_logs, x='confidence', bins=20,
                             title="Response Confidence Scores")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent conversations
        st.subheader("Recent Conversations")
        recent_df = df_logs.tail(10)[['timestamp', 'user_text', 'bot_reply', 'confidence']].copy()
        recent_df['confidence'] = recent_df['confidence'].round(2)
        st.dataframe(recent_df, use_container_width=True)
    
    # Feedback analytics
    if feedback_data:
        st.subheader("User Satisfaction")
        df_feedback = pd.DataFrame(feedback_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Ratings over time
            df_feedback['timestamp'] = pd.to_datetime(df_feedback['timestamp'])
            fig = px.line(df_feedback, x='timestamp', y='rating',
                         title="Ratings Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rating distribution
            rating_counts = df_feedback['rating'].value_counts().sort_index()
            fig = px.bar(x=rating_counts.index, y=rating_counts.values,
                        title="Rating Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Average rating with trend
        avg_rating = df_feedback['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}/5.0", 
                 delta=f"{avg_rating-4:.2f}" if avg_rating != 4 else None)

# =============================
# Enhanced Main Application
# =============================
def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="🎓 University FAQ Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session
    initialize_session()
    
    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://via.placeholder.com/100x100?text=🎓", width=80)
    with col2:
        st.title("🎓 Enhanced University FAQ Chatbot")
        st.caption("Advanced Multilingual Support: English • 中文 • Bahasa Malaysia • தமிழ்")
    
    # Sidebar
    with st.sidebar:
        st.subheader("ℹ️ Information")
        st.info("""
        **I can help you with:**
        • 🎓 Course Information
        • 📝 Admissions Process
        • 💰 Tuition & Scholarships
        • 📚 Academic Schedules
        • 🏠 Campus Facilities
        • ⏰ Office Hours
        """)
        
        st.subheader("🗣️ Language Settings")
        selected_lang = st.selectbox(
            "Preferred Language",
            options=list(SUPPORTED_LANGS.keys()),
            format_func=lambda x: SUPPORTED_LANGS[x],
            key="language_selector"
        )
        st.session_state.user_language = selected_lang
        
        st.subheader("📊 Session Info")
        st.text(f"Session ID: {st.session_state.session_id}")
        st.text(f"Messages: {len(st.session_state.history)}")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.history = []
            st.session_state.course_pending = None
            st.session_state.current_course = None
            st.session_state.conversation_context = []
            st.rerun()
    
    # Enhanced CSS styling
    st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 400px;
        overflow-y: auto;
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        scrollbar-width: thin;
        scrollbar-color: #888 #f1f1f1;
    }
    
    .user-bubble, .bot-bubble {
        padding: 12px 16px;
        border-radius: 20px;
        margin: 8px;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        align-self: flex-end;
        margin-left: auto;
    }
    
    .bot-bubble {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        align-self: flex-start;
        margin-right: auto;
    }
    
    .quick-button {
        margin: 5px;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .quick-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "📊 Analytics", "🔧 Settings"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        show_enhanced_analytics()
    
    with tab3:
        settings_interface()

def chat_interface():
    """Enhanced chat interface"""
    st.subheader("💬 Chat with our AI Assistant")
    
    # Quick action buttons
    st.write("**Quick Questions:**")
    quick_options = [
        "What courses do you offer?",
        "Admission requirements",
        "Tuition fees",
        "Campus facilities",
        "Scholarship information",
        "Academic calendar"
    ]
    
    cols = st.columns(3)
    for i, option in enumerate(quick_options):
        with cols[i % 3]:
            if st.button(option, key=f"quick_{i}", use_container_width=True):
                process_user_input(option)
    
    # Chat display
    display_chat()
    
    # Input area
    user_input = st.text_input(
        "Type your message here...",
        key="chat_input",
        placeholder="Ask me about courses, admissions, fees, or any other university information..."
    )
    
    if user_input:
        process_user_input(user_input)
    
    # Voice input option (placeholder)
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🎤 Voice Input", help="Voice input feature (coming soon)"):
            st.info("Voice input feature will be available in the next update!")

def display_chat():
    """Display chat history with enhanced formatting"""
    if not st.session_state.history:
        st.info("👋 Welcome! Start a conversation by asking about our university programs, admissions, or any other questions you might have.")
        return
    
    chat_html = '<div class="chat-container" id="chat-box">'
    
    for i, (speaker, message) in enumerate(st.session_state.history):
        bubble_class = "user-bubble" if speaker == "You" else "bot-bubble"
        icon = "👤" if speaker == "You" else "🤖"
        
        # Add timestamps for recent messages
        if i >= len(st.session_state.history) - 6:
            timestamp = datetime.datetime.now().strftime("%H:%M")
            chat_html += f'<div class="{bubble_class}">'
            chat_html += f'<small style="opacity:0.7">{icon} {speaker} - {timestamp}</small><br>'
            chat_html += f'{message}</div>'
        else:
            chat_html += f'<div class="{bubble_class}">{icon} {speaker}: {message}</div>'
    
    chat_html += '</div>'
    chat_html += """
    <script>
    setTimeout(function() {
        var chatBox = document.getElementById('chat-box');
        if(chatBox) {
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    }, 100);
    </script>
    """
    
    st.markdown(chat_html, unsafe_allow_html=True)

def process_user_input(user_input):
    """Process user input and generate response"""
    try:
        if not user_input.strip():
            return
        
        # Show typing indicator
        with st.spinner("🤖 Thinking..."):
            # Detect language
            detected_lang = safe_detect_language(user_input)
            
            # Generate response
            response, confidence, translated_input, intent = get_enhanced_response(user_input, detected_lang)
            
            # Add to conversation history
            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("Bot", response))
            
            # Update conversation context
            st.session_state.conversation_context.append({
                "user_input": user_input,
                "intent": intent,
                "confidence": confidence
            })
            
            # Keep only last 10 context items
            if len(st.session_state.conversation_context) > 10:
                st.session_state.conversation_context = st.session_state.conversation_context[-10:]
            
            # Log interaction
            log_interaction(user_input, detected_lang, translated_input, response, confidence, intent)
        
        # Rerun to update chat display
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing your message: {str(e)}")
        st.session_state.history.append(("Bot", "I apologize, but I encountered an error. Please try again."))

def settings_interface():
    """Settings and configuration interface"""
    st.subheader("🔧 System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Chat Settings**")
        
        # Response confidence threshold
        confidence_threshold = st.slider(
            "Response Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Minimum confidence score for responses"
        )
        
        # Language detection sensitivity
        lang_sensitivity = st.selectbox(
            "Language Detection Sensitivity",
            options=["High", "Medium", "Low"],
            index=1,
            help="How strictly to detect language changes"
        )
        
        # Maximum message length
        max_length = st.number_input(
            "Maximum Message Length",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Maximum characters allowed in user messages"
        )
    
    with col2:
        st.write("**Data Management**")
        
        # Export data options
        if st.button("📤 Export Chat Logs"):
            if LOG_PATH.exists():
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    logs = json.load(f)
                st.download_button(
                    "Download Chat Logs",
                    data=json.dumps(logs, indent=2, ensure_ascii=False),
                    file_name=f"chat_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        if st.button("📤 Export Feedback"):
            if FEEDBACK_PATH.exists():
                with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                    feedback = json.load(f)
                st.download_button(
                    "Download Feedback Data",
                    data=json.dumps(feedback, indent=2, ensure_ascii=False),
                    file_name=f"feedback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Clear data options
        st.write("**⚠️ Data Cleanup**")
        if st.button("🗑️ Clear All Logs", type="secondary"):
            if st.checkbox("I confirm I want to delete all chat logs"):
                try:
                    if LOG_PATH.exists():
                        LOG_PATH.unlink()
                    st.success("All chat logs cleared!")
                except Exception as e:
                    st.error(f"Error clearing logs: {e}")
    
    # System information
    st.subheader("📊 System Information")
    
    # File status
    file_status = {
        "Knowledge Base": KB_PATH.exists(),
        "Chat Logs": LOG_PATH.exists(),
        "Feedback Data": FEEDBACK_PATH.exists()
    }
    
    for file_name, exists in file_status.items():
        status_icon = "✅" if exists else "❌"
        st.write(f"{status_icon} {file_name}: {'Available' if exists else 'Not found'}")
    
    # Performance metrics
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
            
            avg_confidence = sum([log.get("confidence", 0) for log in logs]) / len(logs) if logs else 0
            successful_responses = sum([1 for log in logs if log.get("confidence", 0) > 0.5])
            success_rate = (successful_responses / len(logs) * 100) if logs else 0
            
            st.write("**Performance Metrics:**")
            st.write(f"• Average Confidence: {avg_confidence:.2f}")
            st.write(f"• Success Rate: {success_rate:.1f}%")
            st.write(f"• Total Interactions: {len(logs)}")
            
        except Exception as e:
            st.error(f"Error loading performance metrics: {e}")

# Enhanced feedback system
def enhanced_feedback_form():
    """Enhanced feedback collection"""
    st.subheader("💡 Help Us Improve")
    
    with st.form("enhanced_feedback"):
        col1, col2 = st.columns(2)
        
        with col1:
            rating = st.select_slider(
                "How satisfied are you with this conversation?",
                options=[1, 2, 3, 4, 5],
                value=5,
                format_func=lambda x: "⭐" * x
            )
            
            response_speed = st.radio(
                "How was the response speed?",
                options=["Too slow", "Just right", "Very fast"],
                index=1
            )
        
        with col2:
            accuracy = st.radio(
                "How accurate were the responses?",
                options=["Very poor", "Poor", "Average", "Good", "Excellent"],
                index=3
            )
            
            ease_of_use = st.radio(
                "How easy was it to use?",
                options=["Very difficult", "Difficult", "Average", "Easy", "Very easy"],
                index=3
            )
        
        # Additional feedback
        suggestions = st.text_area(
            "Any suggestions for improvement?",
            placeholder="Tell us what we can do better..."
        )
        
        would_recommend = st.checkbox("Would you recommend this chatbot to others?")
        
        if st.form_submit_button("📤 Submit Feedback", type="primary"):
            feedback_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": st.session_state.session_id,
                "overall_rating": rating,
                "response_speed": response_speed,
                "accuracy": accuracy,
                "ease_of_use": ease_of_use,
                "suggestions": suggestions,
                "would_recommend": would_recommend,
                "conversation_length": len(st.session_state.history),
                "languages_used": list(set([log.get("detected_lang", "en") for log in st.session_state.conversation_context]))
            }
            
            save_enhanced_feedback(feedback_data)

def save_enhanced_feedback(feedback_data):
    """Save enhanced feedback data"""
    try:
        feedback_list = []
        if FEEDBACK_PATH.exists():
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
                feedback_list = json.load(f)
        
        feedback_list.append(feedback_data)
        
        with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
            json.dump(feedback_list, f, indent=2, ensure_ascii=False)
        
        st.success("🙏 Thank you for your valuable feedback!")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")

# Run the application
if __name__ == "__main__":
    try:
        knowledge_base = load_knowledge_base()
        main()
        
        # Add feedback form at the bottom
        if len(st.session_state.history) > 0:
            with st.expander("💬 Share Your Feedback", expanded=False):
                enhanced_feedback_form()
                
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")
