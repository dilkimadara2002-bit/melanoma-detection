#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import mysql.connector
from mysql.connector import Error

# --- 1. Load the Model ---
MODEL_PATH = "melanoma_cnn_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please ensure 'melanoma_cnn_model.h5' is in the same directory.")
    model = None

# --- 2. MySQL Connection Function ---
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",      # WAMP usually localhost
            user="root",           # your MySQL username
            password="",           # your MySQL password (blank for WAMP default)
            database="melanoma_app"
        )
        return connection
    except Error as e:
        st.error(f"MySQL connection error: {e}")
        return None

# --- 3. Recommendations Dictionary ---
recommendations = {
    "melanoma": {
        "text": "Based on the prediction, this lesion has characteristics of a melanoma. It is **strongly recommended** that you consult with a dermatologist.",
        "action": "Immediate consultation with a dermatologist is advised."
    },
    "benign": {
        "text": "The analysis suggests this lesion is likely benign. Monitor regularly for changes.",
        "action": "Continue self-monitoring. Consult a doctor if any changes occur."
    }
}

# --- 4. Preprocessing Function ---
def preprocess_image(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# --- 5. Streamlit UI ---
st.set_page_config(page_title="🩺 Melanoma Skin Cancer Detection", layout="centered")
st.title("🩺 Melanoma Skin Cancer Detection")
st.write("Upload a skin lesion image to classify it as **Melanoma** or **Benign**")
st.write("This web page every inputs are saved automtically!")
st.markdown("---")

# --- User Info Form ---
st.subheader("👤 Enter Your Information")
st.write("if you want image predict you must add your information")
name = st.text_input("Full Name")
email = st.text_input("Email")
Role = st.text_input("Role[Doctor/Patient/Researcher]")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        if model is None:
            st.error("Model not available.")
        elif not name or not email:
            st.warning("Please enter your Name and Email before predicting.")
        else:
            with st.spinner("Analyzing image..."):
                try:
                    img = Image.open(io.BytesIO(uploaded_file.read()))
                    processed_img = preprocess_image(img)
                    prediction_value = model.predict(processed_img)[0][0]

                    if prediction_value > 0.5:
                        label = "Melanoma"
                        confidence = float(prediction_value)
                        recom_text = recommendations["melanoma"]["text"]
                        recom_action = recommendations["melanoma"]["action"]
                    else:
                        label = "Benign"
                        confidence = float(1 - prediction_value)
                        recom_text = recommendations["benign"]["text"]
                        recom_action = recommendations["benign"]["action"]

                    # Show Results
                    st.markdown("### Analysis Results")
                    st.info(f"**Prediction:** {label}")
                    st.info(f"**Confidence:** {confidence * 100:.2f}%")
                    st.warning(f"**Recommendation:** {recom_text}")
                    st.error(f"**Clinical Guideline:** {recom_action}")

                    # --- Save to Database ---
                    conn = get_db_connection()
                    if conn:
                        cursor = conn.cursor()

                        # Insert/Find user
                        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
                        user = cursor.fetchone()
                        if user:
                            user_id = user[0]
                        else:
                            cursor.execute(
                                "INSERT INTO users (name, email, Role) VALUES (%s, %s, %s)",
                                (name, email, Role)
                            )
                            conn.commit()
                            user_id = cursor.lastrowid

                        # Insert prediction
                        cursor.execute(
                            "INSERT INTO predictions (user_id, filename, prediction, confidence, recommendation, clinical_guideline) VALUES (%s, %s, %s, %s, %s, %s)",
                            (user_id, uploaded_file.name, label, confidence, recom_text, recom_action)
                        )
                        conn.commit()
                        cursor.close()
                        conn.close()
                        st.success("✅ Prediction saved to database!")

                except Exception as e:
                    st.error(f"Error during prediction: {e}")


# --- 4. Feedback Form UI ---
st.header("📝 Feedback Form")
st.write("Please share your thoughts on the web page and provide any recommendations for improvement.")

with st.form("feedback_form"):
    # Star rating mapping
    rating_options = {
        "⭐": 1,
        "⭐⭐": 2,
        "⭐⭐⭐": 3,
        "⭐⭐⭐⭐": 4,
        "⭐⭐⭐⭐⭐": 5
    }
    
    selected_rating_emoji = st.radio(
        "Overall Rating",
        options=list(rating_options.keys()),
        index=4,  
        horizontal=True,
        help="Rate the app from 1 (poor) to 5 (excellent)."
    )
    
    feedback_text = st.text_area("Your Feedback", help="What did you like or dislike about the app? Any issues?")
    
    submit_button = st.form_submit_button(label="Submit Feedback")

    if submit_button:
        if not email:  # check if user email is entered earlier
            st.warning("⚠️ Please enter your Email in the User Info section before submitting feedback.")
        else:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                # find user by email (must exist because prediction saves user info earlier)
                cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
                user = cursor.fetchone()

                if user:
                    user_id = user[0]
                    cursor.execute(
                        "INSERT INTO feedback (user_id, rating, comments) VALUES (%s, %s, %s)",
                        (user_id, rating_options[selected_rating_emoji], feedback_text)
                    )
                    conn.commit()
                    st.success("✅ Feedback saved. Thank you!")
                else:
                    st.error("⚠️ User not found. Please submit a prediction first before feedback.")

                cursor.close()
                conn.close()

