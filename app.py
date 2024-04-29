import streamlit as st
import requests
import base64

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def predict_label(text):
    response = requests.post("http://localhost:8000/predict", json={"text": text})
    return response.json()["label"]

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1446776653964-20c1d3a81b06");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Main Streamlit code
st.sidebar.title("Fake News Detection App")
st.sidebar.write(
    """
    
    Welcome to the Fake News Detection App! This application is designed to help users identify whether a piece of news is genuine or fake. Using a sophisticated machine learning model, the app analyzes the input text and provides a prediction on its authenticity.
    
    **Features:**
    
    - **Fake News Detection:** Enter a news article or snippet into the app, and it will analyze the content to determine whether it's likely to be true or false.
    
    - **Prediction Results:** The app returns the predicted label, indicating whether the input text is classified as real or fake news.
    
    - **User-Friendly Interface:** With a simple and intuitive interface, users can easily input text and view the prediction results.
    
    **How to Use:**
    
    1. Enter the text of the news article or snippet into the input field.
    2. Click the "Predict" button to analyze the input text.
    3. View the prediction result displayed on the screen.
    
    **GitHub Repository:**
    
    For more details about the model and to contribute to the project, check out the GitHub repository [here](https://github.com/mfarooq0001/fake-news-detection-with-streamlit-frontend).
    """
)

real_gif = "GIFs/real_news.gif"
fake_gif = "GIFs/fake_news.gif"

# Title
st.title("Fake News Detection")

# Text input field
text_input = st.text_area("Enter the news text:", "")

# Predict button
if st.button("Predict"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label = predict_label(text_input)[1]
        if label == "1":
            st.image(real_gif, use_column_width=True)
        else:
            st.image(fake_gif, use_column_width=True)

