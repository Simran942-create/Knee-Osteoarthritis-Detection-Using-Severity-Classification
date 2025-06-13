import streamlit as st
import sys
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Add the scripts directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

# Import necessary modules
try:
    import scripts.db_utils as db_utils
    from about_koa import about_koa
    from doctor_consultation_treatment import doctor_consultation_treatment
    from generate_report import generate_report
    from chatbot import get_chatbot_response
except ModuleNotFoundError as e:
    st.error(f"⚠️ Module import error: {e}. Please check your project directory and scripts.")

# Ensure temp folder exists
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Load model once to prevent multiple reloads
@st.cache_resource
def load_model_once():
    return load_model("trained_model.h5")

model = load_model_once()

# Define severity categories and comments
categories = ["Normal", "Doubtful", "Mild", "Moderate", "Severe"]
comments = {
    "Normal": "✅ Your knee shows no signs of osteoarthritis.",
    "Doubtful": "⚠️ There are minimal changes, which could be early signs of osteoarthritis.",
    "Mild": "⚠️ Small bone spurs are present, suggesting mild osteoarthritis.",
    "Moderate": "⚠️ Noticeable bone spurs and joint space narrowing indicate moderate osteoarthritis.",
    "Severe": "🔥 Significant joint damage and narrowing indicate severe osteoarthritis."
}

# Patient info upload and prediction
def patient_info_upload():
    st.header(" Patient Information & X-ray Upload")

    if "patient_data" not in st.session_state:
        st.session_state["patient_data"] = {}

    st.subheader("📝 Step 1: Enter Patient Information")
    full_name = st.text_input("Full Name", value=st.session_state["patient_data"].get("full_name", ""))
    age = st.number_input("Age", min_value=0, max_value=120, step=1, value=st.session_state["patient_data"].get("age", 0))
    gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"], index=["Select", "Male", "Female", "Other"].index(st.session_state["patient_data"].get("gender", "Select")))
    contact_number = st.text_input("Contact Number", value=st.session_state["patient_data"].get("contact_number", ""))
    symptoms = st.text_area("Symptoms", placeholder="e.g., knee pain, stiffness", value=st.session_state["patient_data"].get("symptoms", ""))
    pain_level = st.slider("Pain Level (1-10)", min_value=1, max_value=10, value=st.session_state["patient_data"].get("pain_level", 5))
    bp_details = st.text_input("Blood Pressure (BP) Details", value=st.session_state["patient_data"].get("bp_details", ""))
    sugar_details = st.text_input("Sugar Details", value=st.session_state["patient_data"].get("sugar_details", ""))

    if st.button("📂 Save Patient Information"):
        if full_name and gender != "Select":
            st.session_state["patient_data"] = {
                "full_name": full_name,
                "age": age,
                "gender": gender,
                "contact_number": contact_number or "Not Provided",
                "symptoms": symptoms or "Not Provided",
                "pain_level": pain_level,
                "bp_details": bp_details or "Not Provided",
                "sugar_details": sugar_details or "Not Provided",
            }
            db_utils.insert_patient_data(st.session_state["patient_data"])
            st.success("✅ Patient information saved successfully!")
        else:
            st.error("⚠️ Please fill in all required fields (Full Name and Gender).")

    if "patient_data" in st.session_state:
        st.write("### 🏥 Saved Patient Information")
        st.json(st.session_state["patient_data"])

    st.subheader("📸 Step 2: Upload X-ray Image")
    uploaded_file = st.file_uploader("Upload an X-ray image (JPEG/PNG format)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 1) / 255.0

            prediction = model.predict(img)
            predicted_class = categories[np.argmax(prediction)]
            comment = comments[predicted_class]

            st.image(file_path, caption='🖼 Uploaded X-ray Image', use_column_width=True)
            st.markdown(
                f"""
                <div style="background-color:#f1fdf6; padding:25px; border-left:10px solid #2e7d32; border-radius:10px; margin-top:20px;">
                    <h2 style="color:#2e7d32; font-size:26px;">🔍 Prediction: <span style="color:#1e88e5;">{predicted_class}</span></h2>
                    <p style="font-size:18px; color:#333;">💬 {comment}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.session_state["patient_data"]["prediction"] = predicted_class
            st.session_state["patient_data"]["comment"] = comment
            db_utils.update_prediction(full_name, predicted_class, comment)
        else:
            st.error("❌ Error loading image. Please upload a valid X-ray.")
# Define custom CSS for background image and styling
def load_custom_css():
    st.markdown(f"""
        <style>
            body {{
                background-image: url("file:///C:/Users/Simran/Downloads/green-background-earth-globe-plant-leaves-some-water-drops-green-background-soft-ambient-light-environmental-314215546.webp");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                font-family: 'Arial', sans-serif;
            }}
            .stApp {{
                background-color: rgba(255, 255, 255, 0.85);
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }}
            h1, h2, h3 {{
                color: #2E7D32;
                text-align: center;
            }}
            .stButton>button {{
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                padding: 10px 15px;
            }}
            .stButton>button:hover {{
                background-color: #45a049;
            }}
            .stSidebar {{
                background-color: rgba(240, 248, 240, 0.9);
                border-right: 2px solid #A5D6A7;
            }}
            .stSidebar h2 {{
                color: #1B5E20;
                text-align: center;
            }}
        </style>
    """, unsafe_allow_html=True)

# Sidebar chatbot functionality
def sidebar_chatbot():
    st.sidebar.title("Chatbot")
    st.sidebar.write("Got questions about Knee Osteoarthritis?")
    user_input = st.sidebar.text_input("Ask your question:")
    if user_input:
        response = get_chatbot_response(user_input)
        st.sidebar.write(f"**Chatbot:** {response}")

# Main homepage content
def front_page():
    st.title("Knee Osteoarthritis Detection")
    st.subheader("AI-Powered Insights for Better Health")

    st.markdown(""" 
    Welcome to our application! This app leverages AI to analyze knee X-rays and provide insights into osteoarthritis severity.
    Early diagnosis is key to improving mobility and quality of life.
    """)

    st.write("### About Knee Osteoarthritis")
    st.write(""" 
    KOA is a degenerative joint disease affecting the cartilage in the knee joint. Symptoms include:
    - Pain during movement.
    - Stiffness, especially after rest.
    - Swelling caused by inflammation.
    """)

    st.info("🔍 Did you know? KOA affects over 10% of men and 13% of women aged 60 and older.")

    st.write("### Quick Navigation")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("📖 Learn About KOA"):
            st.session_state["page_number"] = 0
    with col2:
        if st.button("🦴 Upload X-Ray"):
            st.session_state["page_number"] = 1
    with col3:
        if st.button("👨‍⚕️ Doctor Consultation"):
            st.session_state["page_number"] = 2
    with col4:
        if st.button("📊 Generate Report"):
            st.session_state["page_number"] = 3

    st.markdown("---")

# Navigation buttons at the bottom
def navigation_buttons():
    max_page_number = 3
    col1, col2, col3 = st.columns([6, 2, 2])
    with col2:
        if st.session_state.get("page_number", 0) < max_page_number:
            if st.button("Next ➡️"):
                st.session_state["page_number"] = min(st.session_state["page_number"] + 1, max_page_number)
    with col1:
        if st.session_state.get("page_number", 0) > 0:
            if st.button("⬅️ Back"):
                st.session_state["page_number"] = max(st.session_state["page_number"] - 1, 0)
    with col3:
        if st.button("Exit"):
            st.warning("Exiting the application...")
            st.session_state.clear()
            st.rerun()

# Main app function
def main():
    load_custom_css()
    if "page_number" not in st.session_state:
        st.session_state["page_number"] = 0

    pages = {
        0: front_page,
        1: patient_info_upload,
        2: doctor_consultation_treatment,
        3: generate_report
    }

    current_page = st.session_state["page_number"]
    if current_page in pages:
        pages[current_page]()
    else:
        st.error("Invalid page number! Please navigate using the buttons.")

    navigation_buttons()
    sidebar_chatbot()

if __name__ == "__main__":
    main()
