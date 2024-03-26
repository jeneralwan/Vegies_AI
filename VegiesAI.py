import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from dotenv import load_dotenv
import openai
import requests

load_dotenv(".env")

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def classify_image(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    top_prediction = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=1)[0][0]
    return top_prediction[1]

def get_vegetable_info(vegetable_name):
    """Fetches information about a vegetable from an API (replace with your logic)"""

    vegetable_info = {
        'name': vegetable_name,
    }
    return vegetable_info

def generate_answer(instruction):
    """Generates an answer for a given instruction using OpenAI's Chat API"""
    chat_completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a user."},
            {"role": "system", "content": "I am a model."},
            {"role": "user", "content": instruction}
        ]
    )

    if chat_completion and chat_completion.choices:
        response = chat_completion.choices[0].message.content
        return response
    else:
        return "Error generating answer."


def main():
    st.title("VEGETABLES IDENTIFIER HELPER")
    uploaded_image = st.file_uploader("Please upload your vegies image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Vegies Image", use_column_width=True)

        if st.button("IDENTIFY"):
            with st.spinner('Identifying...'):
                try:
                    vegetable_name = classify_image(image)
                    st.success("Predicted Vegetable: {}".format(vegetable_name))
                except Exception as e:
                    print(f"Error during classification: {e}")
                    st.error(f"An error occurred during classification. Please try again later.")

                # AI-generated answer
                st.write("AI's Response:")
                ai_answer = generate_answer(f"What are the vitamins in {vegetable_name}?")
                st.write(ai_answer)

if __name__ == "__main__":
    main()
