import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="CPredict", page_icon="📊", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://wallpaperaccess.com/full/1102011.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }
    .block-container {
        backdrop-filter: blur(6px);
        background-color: rgba(0, 0, 0, 0.55);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        color: white;
    }
    .centered-title {
        text-align: center;
        color: white;
        font-size: 2.2rem;
        font-weight: bold;
    }
    .tagline {
        text-align: center;
        font-size: 1rem;
        color: #cccccc;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

model = load("rating_model.pkl")

st.markdown("<div class='block-container'>", unsafe_allow_html=True)

st.markdown("<div class='centered-title'>📊 Welcome to <span style='color:#00BFFF;'>CPredict</span></div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>Predict your Codeforces rating change using Machine Learning</div>", unsafe_allow_html=True)

st.markdown("""
<details>
<summary style="font-weight: bold; color: #00BFFF;">ℹ️ How It Works</summary>
<p style="color: white;">
This app uses a <b>Random Forest Regressor</b> trained on past Codeforces contests to estimate your rating change based on your <b>current rating</b> and <b>contest rank</b>. While it's not official, it gives a close approximation to help you set expectations!
</p>
</details>
""", unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        rank = st.number_input("🔢 Your Contest Rank", min_value=1, max_value=100000, value=1000, step=1)
    with col2:
        old_rating = st.number_input("⭐ Your Current Rating", min_value=0, max_value=4000, value=1400, step=10)
    submit = st.form_submit_button("🚀 Predict")

if submit:
    input_df = pd.DataFrame([[old_rating, rank]], columns=["oldRating", "rank"])
    predicted_change = int(model.predict(input_df)[0])
    sign = "+" if predicted_change > 0 else ""
    new_rating = old_rating + predicted_change

    st.success(f"🎯 Predicted Rating Change: {sign}{predicted_change}")
    st.info(f"📈 Estimated New Rating: {new_rating}")

st.markdown("---")
st.markdown("""
<small style='display: block; text-align: center; color: lightgray;'>
Powered by a Random Forest ML model | Built with ❤️ by <a href="https://www.linkedin.com/in/singh-aman23" target="_blank" style="color:#00BFFF; text-decoration: none;"><b>Aman</b></a>
</small>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  
