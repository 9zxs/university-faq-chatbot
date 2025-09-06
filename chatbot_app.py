import streamlit as st
import random
import json
import datetime
import uuid
import pandas as pd
from pathlib import Path
from googletrans import Translator
import joblib

# ==== Paths ====
LOG_PATH = Path("chat_logs.json")

# ==== Load trained model and data ====
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

translator = Translator()

# ==== Helper: Log user-bot interaction ====
def log_interaction(user_text, detected_lang, translated_input, predicted_tag, bot_reply, confidence):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "session_id": st.session_state.session_id,
        "user_text": user_text,
        "detected_lang": detected_lang,
        "translated_input": translated_input,
        "predicted_tag": predicted_tag,
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

# ==== Predict intent ====
def predict_intent(user_text):
    X = vectorizer.transform([user_text])
    probs = model.predict_proba(X)[0]
    max_idx = probs.argmax()
    confidence = probs[max_idx]
    predicted_tag = model.classes_[max_idx]
    return predicted_tag, confidence

# ==== Chatbot response ====
def get_response(user_text):
    detected_lang = translator.detect(user_text).lang

    # Translate input to English for intent classification
    translated_input = translator.translate(user_text, src=detected_lang, dest="en").text
    predicted_tag, confidence = predict_intent(translated_input)

    # Default fallback
    response = "Sorry, I didn't quite get that. Could you rephrase?"

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            break

    # Translate bot response back to detected language
    bot_reply = translator.translate(response, src="en", dest=detected_lang).text

    # Save interaction
    log_interaction(user_text, detected_lang, translated_input, predicted_tag, bot_reply, float(confidence))

    return bot_reply

# ==== UI ====
st.set_page_config(page_title="University Chatbot", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

tabs = st.tabs(["💬 Chat", "📊 Analytics"])

# ==== Tab 1: Chat ====
with tabs[0]:
    st.header("🎓 University FAQ Chatbot")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        response = get_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

# ==== Tab 2: Analytics ====
with tabs[1]:
    st.header("📊 Chat Analytics")

    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)

        if logs:
            df = pd.DataFrame(logs)
            st.dataframe(df, use_container_width=True)

            # ==== Export to CSV ====
            st.subheader("📂 Export Logs")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="chat_logs.csv",
                mime="text/csv"
            )
        else:
            st.info("No logs available yet.")
    else:
        st.info("No logs found.")
