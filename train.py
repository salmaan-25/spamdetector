# train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")




def main():
    print("⬇️ Loading dataset from URL ...")
    df = pd.read_table(DATA_URL, header=None, names=["label", "message"])
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})


    X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
    )


    print("🔠 Vectorizing ...")
    vectorizer = CountVectorizer()
    X_train_dtm = vectorizer.fit_transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)


    print("🧠 Training model ...")
    model = MultinomialNB()
    model.fit(X_train_dtm, y_train)


    print("✅ Evaluating ...")
    y_pred = model.predict(X_test_dtm)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    print(f"💾 Saved model to: {MODEL_PATH}")
    print(f"💾 Saved vectorizer to: {VEC_PATH}")




if __name__ == "__main__":
    main()