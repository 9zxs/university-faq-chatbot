from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib, pandas as pd, json

DATA_PATH = Path(__file__).resolve().parent / "data" / "intents.json"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
REPORT_DIR = Path(__file__).resolve().parent / "reports"

def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)["intents"]
    texts, labels = [], []
    for intent in data:
        tag = intent.get("tag") or intent.get("intent")
        patterns = intent.get("patterns") or intent.get("text") or []
        for p in patterns:
            texts.append(p)
            labels.append(tag)
    return texts, labels

def main():
    texts, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ("svm", LinearSVC())
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(REPORT_DIR / "predictions.csv", index=False)
    joblib.dump(clf, MODEL_PATH)
    print("Model saved to", MODEL_PATH)

if __name__ == "__main__":
    main()
