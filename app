import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Title
st.title("📰 Fake News Detector")
st.write("Enter a news headline or article content to check if it's **REAL** or **FAKE**.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\ATCHAYA\OneDrive\Desktop\Pooja\FakeorReal.csv")
    return df

df = load_data()

# Preprocess and train model
@st.cache_resource
def train_model(df):
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))
    return model, vectorizer, accuracy

model, vectorizer, accuracy = train_model(df)

# Display accuracy
st.success(f"✅ Model trained with {accuracy * 100:.2f}% accuracy.")

# User input
user_input = st.text_area("🔤 Enter news text here:", height=200)

if user_input:
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    st.subheader("🔎 Prediction:")
    if prediction.lower() == 'fake':
        st.error("❌ FAKE News Detected!")
    else:
        st.success("✅ This appears to be REAL News.")

st.markdown("---")
st.caption("Built by Batch 4")
