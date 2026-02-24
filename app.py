# =========================================================
# FOOD CALORIE & FITNESS SYSTEM – PREMIUM SIDEBAR UI
# WITH LOGIN, PERSONALIZATION & WEEKLY REPORT
# =========================================================

import os
import streamlit as st
import numpy as np
import pandas as pd
import json
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
import sqlite3
import datetime
import bcrypt
import plotly.express as px
import streamlit.components.v1 as components

# ---------------- COMMON FOODS & EXERCISE DATA ----------------
COMMON_FOODS = {
    "Idli (2 pcs)": {"calories": 120, "exercise": "Walking – 30 min"},
    "Dosa (Plain)": {"calories": 170, "exercise": "Jogging – 20 min"},
    "Chapati (1)": {"calories": 120, "exercise": "Yoga – 40 min"},
    "Rice (1 cup)": {"calories": 200, "exercise": "Walking – 50 min"},
    "Biryani (1 plate)": {"calories": 450, "exercise": "Running – 45 min"},
    "Poori (2)": {"calories": 280, "exercise": "Skipping – 25 min"},
    "Upma (1 cup)": {"calories": 180, "exercise": "Cycling – 25 min"},
    "Vada (1)": {"calories": 140, "exercise": "Jogging – 15 min"}
}

# ---------------- BASIC SETUP ----------------


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "food_category_model.keras")
LABELS_PATH = os.path.join(BASE_DIR, "class_labels.json")
CALORIES_PATH = os.path.join(BASE_DIR, "food_calories.csv")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DESIGNING FOOD CALORIE ESTIMATION AND FITNESS RECOMMENDATION SYSTEM BASED ON USER INFORMATION",
    page_icon="🍎",
    layout="wide"
)



# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password BLOB,
    age INTEGER,
    gender TEXT,
    height REAL,
    weight REAL,
    activity TEXT,
    goal TEXT,
    diabetes INTEGER,
    acidity INTEGER,
    constipation INTEGER,
    obesity INTEGER
)
""")

conn.commit()



cursor.execute("""
CREATE TABLE IF NOT EXISTS food_logs (
    username TEXT,
    food TEXT,
    calories REAL,
    date TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS weight_logs (
    username TEXT,
    weight REAL,
    date TEXT
)
""")
cursor.execute("""
CREATE TABLE IF NOT EXISTS newsletter_subscribers (
    email TEXT PRIMARY KEY,
    date TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS exercise_logs (
    username TEXT,
    exercise TEXT,
    minutes REAL,
    calories_burned REAL,
    date TEXT
)
""")
conn.commit()



# ---------------- HELPERS ----------------
def hash_password(password):
    # Generate salt + hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def bmi_calc(weight, height_cm):
    h = height_cm / 100
    return weight / (h ** 2)

def clean_dataframe_for_streamlit(df):
    def clean_value(val):
        if isinstance(val, bytes):
            # Decode Windows-encoded bytes safely
            return val.decode("cp1252", errors="ignore")
        if pd.isna(val):
            return ""
        try:
            return str(val)
        except Exception:
            return ""

    for col in df.columns:
        df[col] = df[col].map(clean_value)

    return df
def get_latest_weight(username, profile_weight):
    row = cursor.execute("""
        SELECT weight FROM weight_logs
        WHERE username=?
        ORDER BY date DESC
        LIMIT 1
    """, (username,)).fetchone()

    return row[0] if row else profile_weight
def calculate_target_calories(age, gender, height, weight, activity, goal):
    if gender == "Female":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5

    activity_factor = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }

    maintenance = bmr * activity_factor.get(activity, 1.2)

    if goal == "Weight Loss":
        target = maintenance - 300
    elif goal == "Weight Gain":
        target = maintenance + 300
    else:
        target = maintenance

    return maintenance, target

def metric_card(label, value, icon=""):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">
                {icon} {label}
            </div>
            <div class="metric-value">
                {value}
            </div>
        </div>
    """, unsafe_allow_html=True)



exercise_map = {
    "Idli": ("Brisk walking", 25),
    "Dosa": ("Cycling", 35),
    "Poori": ("Jogging", 45),
    "Biryani": ("Running", 60),
    "Rice": ("Walking", 30),
    "Curd Rice": ("Yoga", 20),
    "Chapati": ("Skipping", 25),
    "Upma": ("Walking", 20),
    "Vada": ("Jogging", 40),
    "Sambar": ("Light stretching", 15),
    "South Indian Thali": ("Mixed cardio", 60)
}


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    K.clear_session()
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

try:
    model = load_model()
except Exception as e:
    model = None
    st.error("⚠️ Model could not be loaded.")
    st.text(str(e))

# ---------------- LOAD LABELS & CALORIE DATA ----------------
with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index → class name
class_names = {v: k for k, v in class_indices.items()}

# Load calorie CSV
calorie_df = pd.read_csv(CALORIES_PATH)



# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "show_result" not in st.session_state:
    st.session_state.show_result = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("""
<style>

/* ================= GLOBAL FONT ================= */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

/* ================= GLOBAL TEXT VISIBILITY ================= */

h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
}

p, span, label, div, small {
    color: #F1F5F9 !important;
}

input, textarea {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* =========================================================
   LOGIN + SIDEBAR GRADIENT (DARK BLUE → CYAN)
========================================================= */

section[data-testid="stSidebar"],
body:has(.login-active) .stApp {
    background: linear-gradient(
        90deg,
        rgba(2, 0, 36, 1) 0%,
        rgba(9, 9, 121, 1) 35%,
        rgba(0, 212, 255, 1) 100%
    ) !important;
}

/* =========================================================
   MAIN PAGE GRADIENT (CYAN → VIOLET VERTICAL)
========================================================= */

body:not(:has(.login-active)) .stApp {
    background: #020024;
    background: -webkit-linear-gradient(90deg, rgba(2, 0, 36, 1) 0%, rgba(9, 9, 121, 1) 100%, rgba(0, 212, 255, 1) 100%);
    background: -moz-linear-gradient(90deg, rgba(2, 0, 36, 1) 0%, rgba(9, 9, 121, 1) 100%, rgba(0, 212, 255, 1) 100%);
    background: linear-gradient(90deg, rgba(2, 0, 36, 1) 0%, rgba(9, 9, 121, 1) 100%, rgba(0, 212, 255, 1) 100%);
    filter: progid:DXImageTransform.Microsoft.gradient(startColorstr="#020024", endColorstr="#00D4FF", GradientType=1);
    background-attachment: fixed;
}

/* =========================================================
   SIDEBAR STYLING
========================================================= */

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,0.25);
}

/* Sidebar Navigation Pills */

section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    border-radius: 999px;
    padding: 10px 16px;
    margin: 6px 0;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.4);
    color: #FFFFFF !important;
    font-weight: 500;
    transition: 0.25s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(255,255,255,0.25);
    transform: translateX(4px);
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(255,255,255,0.35);
    color: #020024 !important;
    font-weight: 600;
}

/* Sidebar Buttons */

section[data-testid="stSidebar"] .stButton > button {
    border-radius: 14px !important;
    background: rgba(255,255,255,0.25) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.4) !important;
    font-weight: 600;
}

/* =========================================================
   LOGIN CARD (DO NOT CHANGE CONTAINER STRUCTURE)
========================================================= */

body:has(.login-active) section[data-testid="stSidebar"] {
    display: none !important;
}

body:has(.login-active) .block-container {
    max-width: 520px !important;
    margin: 12vh auto !important;
    padding: 60px !important;
    border-radius: 28px;

    background: rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(18px);

    border: 1px solid rgba(255,255,255,0.4);

    box-shadow:
        0 40px 80px rgba(0,0,0,0.6);
}

/* Login Inputs */

body:has(.login-active) div[data-baseweb="input"] > div {
    border-radius: 14px !important;
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.5) !important;
}

body:has(.login-active) input {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

body:has(.login-active) label {
    color: #FFFFFF !important;
}

body:has(.login-active) .stButton > button {
    border-radius: 14px !important;
    background: rgba(255,255,255,0.25) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.5) !important;
    font-weight: 600;
}

/* =========================================================
   MOBILE RESPONSIVENESS
========================================================= */

@media (max-width: 768px) {
    .block-container {
        padding: 30px !important;
    }
}

</style>
""", unsafe_allow_html=True)

if not st.session_state.get("logged_in", False):

    st.markdown('<div class="login-active"></div>', unsafe_allow_html=True)

    st.markdown("<h2>Food Fitness</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;opacity:0.7;color:#e0c7ff;'>AI Nutrition Intelligence Platform</p>", unsafe_allow_html=True)

    auth_mode = st.radio(
        "",
        ["Login", "Sign Up"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= LOGIN =================
    if auth_mode == "Login":

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Login", use_container_width=True):

            if not username or not password:
                st.error("Enter username and password")
            else:
                cursor.execute("SELECT password FROM users WHERE username=?", (username,))
                result = cursor.fetchone()

                if result and bcrypt.checkpw(password.encode("utf-8"), result[0]):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    # ================= SIGNUP =================
    else:

        su_username = st.text_input("Username")
        su_password = st.text_input("Password", type="password")
        su_confirm = st.text_input("Confirm Password", type="password")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            su_age = st.number_input("Age", 10, 100, 22)
            su_height = st.number_input("Height (cm)", 120, 220, 160)

        with col2:
            su_weight = st.number_input("Weight (kg)", 30, 150, 55)
            su_gender = st.selectbox("Gender", ["Female", "Male"])

        su_activity = st.selectbox(
            "Activity Level",
            ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Create Account", use_container_width=True):

            if not su_username or not su_password:
                st.error("Username & password required")

            elif len(su_password) < 6:
                st.error("Password must be at least 6 characters")

            elif su_password != su_confirm:
                st.error("Passwords do not match")

            else:
                cursor.execute("SELECT username FROM users WHERE username=?", (su_username,))
                if cursor.fetchone():
                    st.error("Username already exists")
                else:
                    hashed_pw = bcrypt.hashpw(
                        su_password.encode("utf-8"),
                        bcrypt.gensalt()
                    )

                    cursor.execute("""
                        INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        su_username,
                        hashed_pw,
                        su_age,
                        su_gender,
                        su_height,
                        su_weight,
                        su_activity,
                        "Maintain",
                        0, 0, 0, 0
                    ))

                    conn.commit()
                    st.success("Account created successfully. Please login.")

    st.stop()

# =========================================================
# SIDEBAR – CLEAN SaaS STYLE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

menu_options = [
    "Home",
    "Analyze Food",
    "Health Insights",
    "Healthy Weight Toolkit",
    "Ingredients Guide",
    "Fight Sugar Cravings",
    "Lose Weight Safely",
    "About",
]

if st.session_state.page not in menu_options:
    st.session_state.page = "Home"

# Premium Brand Card
st.sidebar.markdown(f"""
<div style="
padding:22px;
border-radius:20px;
background:linear-gradient(135deg,#4C1D95,#6D28D9);
color:white;
margin-bottom:24px;
">
    <h3 style="margin:0;font-weight:600;">Food Fitness</h3>
    <p style="margin:6px 0 12px 0;font-size:12px;opacity:0.85;">
        AI Nutrition Intelligence
    </p>
    <small>Welcome,<br><b>{st.session_state.username}</b></small>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<small style='opacity:0.5;'>NAVIGATION</small>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    menu_options,
    key="sidebar_nav"
)

st.session_state.page = page

st.sidebar.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# Logout Button
st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
logout = st.sidebar.button("Logout", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if logout:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.page = "Home"
    st.rerun()

# =========================================================
# HOME – STARTUP DASHBOARD
# =========================================================
if st.session_state.page == "Home":

    # ---------------- HERO ----------------
    st.markdown("""
    <div class="hero-card">
        <h1>Welcome Back</h1>
        <p>
            Your personalized nutrition intelligence dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # ---------------- LOAD PROFILE ----------------
    cursor.execute("""
    SELECT age, gender, height, weight, activity, goal
    FROM users WHERE username=?
    """, (st.session_state.username,))
    profile = cursor.fetchone()

    if profile and all(profile):

        age, gender, height, weight, activity, goal = profile
        bmi = bmi_calc(weight, height)

        # ---------------- SNAPSHOT ----------------
        st.markdown("""
        <div class="page-header">
            <h1>Health Snapshot</h1>
            <p>Real-time overview of your current status</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Weight", f"{weight} kg", "⚖️")

        with c2:
            metric_card("BMI", f"{bmi:.2f}", "📏")

        with c3:
            metric_card("Activity", activity, "🏃")

        with c4:
            metric_card("Goal", goal, "🎯")

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # ---------------- SMART INSIGHT ----------------
        st.markdown("""
        <div class="card">
            <h3>AI Insight</h3>
        """, unsafe_allow_html=True)

        if bmi < 18.5:
            st.info("You are currently underweight. Focus on gradual calorie surplus and strength training.")
        elif bmi < 25:
            st.success("You are within a healthy BMI range. Maintain consistency.")
        elif bmi < 30:
            st.warning("You are slightly above ideal range. Moderate calorie control recommended.")
        else:
            st.error("High BMI detected. Structured fat-loss strategy advised.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        # ---------------- QUICK ACTIONS ----------------
        st.markdown("""
        <div class="page-header">
            <h1>Quick Actions</h1>
            <p>Start tracking or analyzing instantly</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("""
            <div class="feature-card">
                <h4>Analyze Food</h4>
                <p>Upload food image or log manually.</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="feature-card">
                <h4>View Insights</h4>
                <p>Check calorie trends & adaptive targets.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Complete your profile in Analyze Food to activate dashboard insights.")

    # ---------------- FEATURES SECTION ----------------
    st.markdown('<div class="section-title">What You Can Do Here</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
        <div class="feature-card">
            <h4>AI Food Analysis</h4>
            <ul>
                <li>Upload food images</li>
                <li>Instant calorie estimates</li>
                <li>AI-powered recognition</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    with f2:
        st.markdown("""
        <div class="feature-card">
            <h4>Health Insights</h4>
            <ul>
                <li>Weekly calorie reports</li>
                <li>Food intake trends</li>
                <li>Visual progress tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    with f3:
        st.markdown("""
        <div class="feature-card">
            <h4>Healthy Weight Toolkit</h4>
            <ul>
                <li>BMI analysis</li>
                <li>Weight predictions</li>
                <li>Smart AI tips</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("---")

    # ---------------- MOTIVATION / TIP CARD ----------------
    st.subheader("💡 Today’s Health Tip")

    tips = [
        "Drink a glass of water before meals to control appetite.",
        "Focus on consistency, not perfection.",
        "Half your plate should be vegetables.",
        "Small daily walks make a big difference.",
        "Sleep is as important as diet for weight control."
    ]

    st.info(f"🌿 {np.random.choice(tips)}")

    st.markdown("---")

    st.caption("👉 Use the sidebar to explore food analysis, insights, and tools")

# =========================================================
# ANALYZE FOOD – STARTUP GRADE UI VERSION
# =========================================================
elif st.session_state.page == "Analyze Food":

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------
    st.markdown("""
    <div class="page-header">
        <h1>AI Food Analysis</h1>
        <p>Upload a food image or select food manually</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- SAFETY CHECK ----------------
    if model is None:
        st.error("AI model not loaded. Please restart the application.")
        st.stop()

    # ---------------- SESSION INIT ----------------
    defaults = {
        "current_image": None,
        "analysis_done": False,
        "top_results": None,
        "selected_food": None
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ---------------- MODE SELECT ----------------
    mode = st.radio(
        "",
        ["Upload Image", "Select Manually"],
        horizontal=True
    )

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # =====================================================
    # IMAGE MODE
    # =====================================================
    if mode == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload Food Image",
            type=["jpg", "jpeg", "png"],
            key=f"food_uploader_{st.session_state.uploader_key}"
        )

        if uploaded_file and st.session_state.current_image is None:
            st.session_state.current_image = Image.open(uploaded_file).convert("RGB")

        if st.session_state.current_image is not None:

            col1, col2 = st.columns([1.2, 1])

            with col1:
                st.image(st.session_state.current_image, use_container_width=True)

            with col2:

                if not st.session_state.analysis_done:
                    if st.button("Analyze Food", use_container_width=True):
                        with st.spinner("Analyzing with AI model..."):
                            processed = preprocess_image(
                                np.array(st.session_state.current_image)
                            )
                            preds = model.predict(processed, verbose=0)[0]

                        top_indices = preds.argsort()[-3:][::-1]

                        st.session_state.top_results = [
                            (class_names[i], float(preds[i] * 100))
                            for i in top_indices
                        ]

                        st.session_state.analysis_done = True
                        st.rerun()

                if st.button("Remove Image", use_container_width=True):
                    st.session_state.current_image = None
                    st.session_state.analysis_done = False
                    st.session_state.top_results = None
                    st.session_state.selected_food = None
                    st.session_state.uploader_key += 1
                    st.rerun()

        # ---------------- SHOW RESULTS ----------------
        if st.session_state.analysis_done and st.session_state.top_results:

            st.markdown("<h3>AI Predictions</h3>", unsafe_allow_html=True)

            for food, conf in st.session_state.top_results:
                st.markdown(f"""
                <div class="feature-card" style="margin-bottom:10px;">
                    <strong>{food}</strong><br>
                    Confidence: {conf:.2f}%
                </div>
                """, unsafe_allow_html=True)

            st.session_state.selected_food = st.selectbox(
                "Confirm the detected food:",
                [food for food, _ in st.session_state.top_results]
            )

            grams = st.slider("Portion Size (grams)", 50, 500, 100, 10)

            row = calorie_df[
                calorie_df["category"] == st.session_state.selected_food
            ]

            if not row.empty:

                calories_per_100g = float(row["calories_per_100g"].values[0])
                total_calories = (calories_per_100g / 100) * grams

                col1, col2 = st.columns(2)
                col1.metric("Portion", f"{grams} g")
                col2.metric("Total Calories", f"{total_calories:.0f} kcal")

                if st.button("Confirm & Log Food", use_container_width=True):

                    today = datetime.date.today().isoformat()

                    cursor.execute(
                        "INSERT INTO food_logs VALUES (?, ?, ?, ?)",
                        (
                            st.session_state.username,
                            st.session_state.selected_food,
                            total_calories,
                            today
                        )
                    )
                    conn.commit()

                    st.success("Food logged successfully.")

                    # Reset
                    st.session_state.current_image = None
                    st.session_state.analysis_done = False
                    st.session_state.top_results = None
                    st.session_state.selected_food = None
                    st.session_state.uploader_key += 1
                    st.rerun()

    # =====================================================
    # MANUAL MODE
    # =====================================================
    else:

        st.markdown("<h3>Manual Food Selection</h3>", unsafe_allow_html=True)

        manual_food = st.selectbox(
            "Select Food Item",
            sorted(calorie_df["category"].unique())
        )

        manual_grams = st.slider(
            "Quantity (grams)",
            50, 500, 100, 10
        )

        manual_row = calorie_df[
            calorie_df["category"] == manual_food
        ]

        if not manual_row.empty:

            calories_per_100g = float(
                manual_row["calories_per_100g"].values[0]
            )

            total_calories = (calories_per_100g / 100) * manual_grams

            col1, col2 = st.columns(2)
            col1.metric("Portion", f"{manual_grams} g")
            col2.metric("Total Calories", f"{total_calories:.0f} kcal")

            if st.button("Log Manual Food", use_container_width=True):

                today = datetime.date.today().isoformat()

                cursor.execute(
                    "INSERT INTO food_logs VALUES (?, ?, ?, ?)",
                    (
                        st.session_state.username,
                        manual_food,
                        total_calories,
                        today
                    )
                )
                conn.commit()

                st.success("Manual food logged successfully.")

    st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# HEALTH INSIGHTS – FULLY UPGRADED VERSION
# =========================================================
elif st.session_state.page == "Health Insights":

    st.title("Health Insights")
    st.caption("Track calories • Log exercise • Monitor balance")

    # ---------------- LOAD USER PROFILE ----------------
    cursor.execute("""
    SELECT age, gender, height, weight, activity, goal
    FROM users WHERE username=?
    """, (st.session_state.username,))
    profile = cursor.fetchone()

    if not profile:
        st.warning("Please complete your profile first.")
        st.stop()

    age, gender, height, profile_weight, activity, goal = profile

    # ---------------- GET LATEST WEIGHT (Dynamic System) ----------------
    latest_weight_row = cursor.execute("""
    SELECT weight FROM weight_logs
    WHERE username=?
    ORDER BY date DESC
    LIMIT 1
    """, (st.session_state.username,)).fetchone()

    weight = get_latest_weight(st.session_state.username, profile_weight)
    maintenance_calories, target_calories = calculate_target_calories(
        age, gender, height, weight, activity, goal
    )


    # ---------------- CALCULATE BMR ----------------
    if gender == "Female":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5

    activity_factor = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }

    maintenance_calories = bmr * activity_factor[activity]

    # ---------------- GOAL BASED TARGET ----------------
    if goal == "Weight Loss":
        target_calories = maintenance_calories - 300
    elif goal == "Weight Gain":
        target_calories = maintenance_calories + 300
    else:
        target_calories = maintenance_calories

    # ==================================================
    # TODAY'S CALORIE SUMMARY
    # ==================================================
    st.subheader("Today's Calorie Summary")

    today = datetime.date.today().isoformat()

    # ----- Food Consumed -----
    df_today = pd.read_sql_query("""
    SELECT calories FROM food_logs
    WHERE username=? AND date=?
    """, conn, params=(st.session_state.username, today))

    consumed_today = df_today["calories"].sum() if not df_today.empty else 0

    # ----- Calories Burned -----
    df_ex_today = pd.read_sql_query("""
    SELECT calories_burned FROM exercise_logs
    WHERE username=? AND date=?
    """, conn, params=(st.session_state.username, today))

    burned_today = df_ex_today["calories_burned"].sum() if not df_ex_today.empty else 0

    # ----- Net Calories -----
    net_calories = consumed_today - burned_today
    remaining_calories = target_calories - net_calories

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Target", f"{target_calories:.0f} kcal", "🎯")

    with col2:
        metric_card("Consumed", f"{consumed_today:.0f} kcal", "🔥")

    with col3:
        metric_card("Burned", f"{burned_today:.0f} kcal", "🏃")

    with col4:
        metric_card("Net", f"{net_calories:.0f} kcal", "⚖️")

    if net_calories > target_calories:
        st.error("⚠️ You are in calorie surplus today.")
    else:
        st.success(" You are within calorie control today.")

    st.markdown("---")

    # ==================================================
    #  EXERCISE LOGGING SECTION
    # ==================================================
    st.subheader("Log Today's Exercise")

    exercise_type = st.selectbox(
        "Select Exercise",
        ["Walking", "Jogging", "Running", "Cycling", "Yoga"]
    )

    minutes = st.number_input(
        "Duration (minutes)",
        min_value=0,
        max_value=300,
        step=5
    )

    MET_VALUES = {
        "Walking": 3.5,
        "Jogging": 7,
        "Running": 11,
        "Cycling": 8,
        "Yoga": 3
    }

    if st.button("➕ Log Exercise"):

        if minutes == 0:
            st.warning("Please enter exercise duration.")
        else:
            met = MET_VALUES[exercise_type]
            hours = minutes / 60
            calories_burned = met * weight * hours

            cursor.execute("""
            INSERT INTO exercise_logs
            (username, exercise, minutes, calories_burned, date)
            VALUES (?, ?, ?, ?, ?)
            """, (
                st.session_state.username,
                exercise_type,
                minutes,
                calories_burned,
                today
            ))

            conn.commit()

            st.success(f"{calories_burned:.0f} kcal burned logged successfully!")
            st.rerun()

    st.markdown("---")
    st.markdown("## Weekly Energy Balance")

    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

    # Fetch food data
    df_food_week = pd.read_sql_query("""
    SELECT date, calories FROM food_logs
    WHERE username=? AND date>=?
    """, conn, params=(st.session_state.username, week_ago))

    # Fetch exercise data
    df_ex_week = pd.read_sql_query("""
    SELECT date, calories_burned FROM exercise_logs
    WHERE username=? AND date>=?
    """, conn, params=(st.session_state.username, week_ago))

    # Prepare daily totals
    food_grouped = df_food_week.groupby("date")["calories"].sum().reset_index()
    ex_grouped = df_ex_week.groupby("date")["calories_burned"].sum().reset_index()

    # Merge both
    merged = pd.merge(food_grouped, ex_grouped, on="date", how="outer").fillna(0)

    merged["net"] = merged["calories"] - merged["calories_burned"]

    import plotly.graph_objects as go

    fig = go.Figure()

    # Consumed bars
    fig.add_trace(go.Bar(
        x=merged["date"],
        y=merged["calories"],
        name="Calories Consumed",
        marker_color="#2563eb"
    ))

    # Burned bars
    fig.add_trace(go.Bar(
        x=merged["date"],
        y=merged["calories_burned"],
        name="Calories Burned",
        marker_color="#16a34a"
    ))

    # Net line
    fig.add_trace(go.Scatter(
        x=merged["date"],
        y=merged["net"],
        name="Net Balance",
        mode="lines+markers",
        line=dict(color="#ef4444", width=3)
    ))

    fig.update_layout(
    barmode="group",
    title="Weekly Energy Balance Overview",
    title_x=0.5,
    xaxis_title="Date",
    yaxis_title="Calories (kcal)",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E5E7EB"),
)

    st.plotly_chart(fig, width="stretch")
    st.markdown("---")
    st.subheader("AI Daily Health Analysis")

    # Average net calories for week
    if not merged.empty:

        weekly_net_avg = merged["net"].mean()
        difference_from_target = weekly_net_avg - target_calories

        st.markdown("### Weekly Behavioral Insight")

        if difference_from_target > 150:
            st.error(
                "You are consistently in calorie surplus. "
                "This may lead to gradual weight gain."
            )

        elif difference_from_target < -150:
            st.success(
                "You are maintaining a calorie deficit. "
                "If consistent, weight reduction is expected."
            )

        else:
            st.info(
                "Your calorie intake is close to maintenance level. "
                "Weight stability is expected."
            )

        predicted_weekly_change = difference_from_target * 7 / 7700

        st.markdown("### Predicted Impact")

        if predicted_weekly_change > 0:
            st.warning(
                f"If this continues, you may gain approx "
                f"{predicted_weekly_change:.2f} kg per week."
            )
        else:
            st.success(
                f"If this continues, you may lose approx "
                f"{abs(predicted_weekly_change):.2f} kg per week."
            )

    else:
        st.info("Not enough weekly data for AI insight.")
    # ==================================================
    # ADAPTIVE TARGET ADJUSTMENT SYSTEM (IMPROVED)
    # ==================================================
    st.markdown("---")
    st.subheader("Adaptive Calorie Adjustment")

    if not merged.empty:

        weekly_net_avg = merged["net"].mean()
        adaptive_target = target_calories
        adjustment_message = None

        # --- WEIGHT LOSS MODE ---
        if goal == "Weight Loss":

            if weekly_net_avg > target_calories + 150:
                adaptive_target = target_calories - 150
                adjustment_message = "Deficit too small. Reducing calories slightly."

            elif weekly_net_avg < target_calories - 400:
                adaptive_target = target_calories + 100
                adjustment_message = "Deficit too aggressive. Increasing calories for sustainability."

            else:
                adjustment_message = "Current calorie target is working well."

        # --- WEIGHT GAIN MODE ---
        elif goal == "Weight Gain":

            if weekly_net_avg < target_calories - 150:
                adaptive_target = target_calories + 150
                adjustment_message = "Surplus too small. Increasing calories."

            elif weekly_net_avg > target_calories + 400:
                adaptive_target = target_calories - 100
                adjustment_message = "Surplus too high. Reducing slightly to limit fat gain."

            else:
                adjustment_message = "Current surplus is appropriate."

        # --- MAINTENANCE MODE ---
        else:

            if abs(weekly_net_avg - target_calories) > 250:
                if weekly_net_avg > target_calories:
                    adaptive_target = target_calories - 100
                else:
                    adaptive_target = target_calories + 100

                adjustment_message = "Adjusting calories slightly to stabilize weight."

            else:
                adjustment_message = "Maintenance intake is balanced."

        # --- DISPLAY ---
        st.metric("Recommended Target", f"{adaptive_target:.0f} kcal/day")

        if adaptive_target != target_calories:
            st.warning(adjustment_message)
        else:
            st.success(adjustment_message)

    else:
        st.info("Adaptive system requires at least a few days of logs.")


elif st.session_state.page == "Facts & Myths":

    st.title("Food & Calories – Myths vs Facts")
    st.caption("Clear the confusion. Eat smart. Stay healthy.")

    st.divider()

    myths_facts = [
        ("Eating less always means losing weight",
         "Extreme calorie restriction slows metabolism and causes fatigue. Balanced nutrition matters."),

        ("All calories are the same",
         "100 calories from vegetables affect the body differently than 100 calories from sugar or junk food."),

        ("Skipping meals helps burn fat",
         "Skipping meals often leads to overeating later and unstable blood sugar levels."),

        ("Carbs make you fat",
         "Complex carbs like rice, oats, fruits, and millets are essential energy sources."),

        ("Fat-free foods are always healthy",
         "Many fat-free foods contain added sugar. Healthy fats are essential for hormones."),

        ("Late-night eating causes weight gain",
         "Total daily calorie intake matters more than meal timing."),

        ("Fruits can be eaten without limits",
         "Fruits are healthy but still contain calories and natural sugars."),

        ("Exercise alone is enough for weight loss",
         "Weight loss is about 70% diet and 30% exercise."),

        ("Drinking water burns fat instantly",
         "Water supports digestion but does not directly burn fat."),

        ("You must avoid favorite foods completely",
         "Sustainable fitness comes from balance, not strict restriction.")
    ]

    col1, col2 = st.columns(2)

    for i, (myth, fact) in enumerate(myths_facts):
        container = col1 if i % 2 == 0 else col2
        with container:
            st.markdown(f"""
                <div style="
                    background: var(--secondary-background-color);
                    padding:18px;
                    border-radius:16px;
                    box-shadow:0 6px 18px rgba(0,0,0,0.08);
                    margin-bottom:16px;
                ">
                    <h4 style="color:#ef4444;">Myth</h4>
                    <p style="font-weight:600;">{myth}</p>
                    <hr>
                    <h4 style="color:#16a34a;">Fact</h4>
                    <p>{fact}</p>
                </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Quick Nutrition Fundamentals")

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        metric_card("Protein", "Muscle repair", "🥦")

    with metric2:
        metric_card("Fiber", "Hunger control", "🥗")

    with metric3:
        metric_card("Healthy Fats", "Hormone balance", "🫒")

    with metric4:
        metric_card("Low Sugar", "Long-term health", "🧂")

    st.info("Consistency beats restriction. Sustainable habits win.")

    st.divider()

    st.subheader("Quiz Mode — Myth or Fact?")

    # Initialize quiz state
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0

    if "quiz_question" not in st.session_state:
        st.session_state.quiz_question = None

    if "quiz_answer" not in st.session_state:
        st.session_state.quiz_answer = None

    if "quiz_feedback" not in st.session_state:
        st.session_state.quiz_feedback = None


    # Create quiz pool
    quiz_pool = [
        ("Eating less always guarantees weight loss", "Myth"),
        ("Total calorie intake matters more than meal timing", "Fact"),
        ("Carbs automatically cause fat gain", "Myth"),
        ("Protein helps in muscle repair", "Fact"),
        ("Drinking water directly burns fat", "Myth"),
        ("Fiber improves digestion and hunger control", "Fact"),
        ("Exercise alone is enough for fat loss", "Myth"),
        ("Healthy fats are essential for hormones", "Fact"),
    ]

    # Generate new question
    import random

    if st.button("🎲 New Question") or st.session_state.quiz_question is None:
        question, answer = random.choice(quiz_pool)
        st.session_state.quiz_question = question
        st.session_state.quiz_answer = answer
        st.session_state.quiz_feedback = None


    # Display question
    if st.session_state.quiz_question:

        st.markdown(f"""
            <div style="
                background: var(--secondary-background-color);
                padding:20px;
                border-radius:18px;
                box-shadow:0 8px 24px rgba(0,0,0,0.08);
                margin-bottom:15px;
            ">
                <h4>{st.session_state.quiz_question}</h4>
            </div>
        """, unsafe_allow_html=True)

        user_choice = st.radio(
            "Your Answer:",
            ["Myth", "Fact"],
            key="quiz_choice",
            horizontal=True
        )

        if st.button("Submit Answer"):

            if user_choice == st.session_state.quiz_answer:
                st.session_state.quiz_score += 1
                st.session_state.quiz_feedback = "correct"
            else:
                st.session_state.quiz_feedback = "wrong"

        # Feedback display
        if st.session_state.quiz_feedback == "correct":
            st.success("Correct! Well done.")

        elif st.session_state.quiz_feedback == "wrong":
            st.error(
                f"Incorrect. Correct answer: {st.session_state.quiz_answer}"
            )

        # Score Display
        st.markdown(f"### Score: {st.session_state.quiz_score}")

        if st.button("Reset Quiz"):
            st.session_state.quiz_score = 0
            st.session_state.quiz_question = None
            st.session_state.quiz_feedback = None
            st.rerun()


elif st.session_state.page == "Healthy Weight Toolkit":

    st.title("Healthy Weight Toolkit")
    st.caption("Personalized insights • Smart predictions • Healthy habits")

    # ---------------- LOAD USER PROFILE ----------------
    cursor.execute("""
    SELECT age, gender, height, weight, activity, goal
    FROM users WHERE username=?
    """, (st.session_state.username,))
    profile = cursor.fetchone()

    if not profile or not all(profile):
        st.warning("Please complete your profile in 'Analyze Food' section.")
        st.stop()

    age, gender, height, profile_weight, activity, goal = profile


    # ---------------- GET LATEST LOGGED WEIGHT ----------------
    latest_weight_row = cursor.execute("""
    SELECT weight FROM weight_logs
    WHERE username=?
    ORDER BY date DESC
    LIMIT 1
    """, (st.session_state.username,)).fetchone()

    if latest_weight_row:
        weight = latest_weight_row[0]   # use latest logged weight
    else:
        weight = profile_weight        # fallback to profile weight


    # ---------------- CALCULATE BMI USING DYNAMIC WEIGHT ----------------
    bmi = bmi_calc(weight, height)


    # ---------------- CARD UI ----------------
    
    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("BMI", f"{bmi:.2f}", "📏")

    with c2:
        metric_card("Weight", f"{weight} kg", "⚖️")

    with c3:
        metric_card("Goal", goal, "🎯")
        
    st.markdown("---")

    # ---------------- BMI STATUS & AI TIPS ----------------
    st.subheader("Smart AI Health Tips")

    if bmi < 18.5:
        status = "Underweight"
        tips = [
            "Increase calorie-dense healthy foods",
            "Add strength training",
            "Avoid skipping meals"
        ]
    elif bmi < 25:
        status = "Healthy Weight"
        tips = [
            "Maintain balanced nutrition",
            "Stay consistent with exercise",
            "Avoid emotional eating"
        ]
    elif bmi < 30:
        status = "Overweight"
        tips = [
            "Reduce sugary & fried foods",
            "Increase daily steps",
            "Focus on portion control"
        ]
    else:
        status = "Obese"
        tips = [
            "Gradual calorie deficit",
            "Low-impact exercises",
            "Consult a nutrition expert if needed"
        ]

    st.success(f"BMI Category: **{status}**")
    for t in tips:
        st.write("•", t)

    st.markdown("---")

    # ---------------- PROGRESS TRACKING ----------------
    st.subheader("Progress Tracking")

    today = datetime.date.today().isoformat()
    # ---------------- WEIGHT INPUT ----------------
    new_weight = st.number_input(
        "Enter Today's Weight (kg)",
        min_value=30.0,
        max_value=200.0,
        step=0.1,
        value=float(weight)
    )

    if st.button("➕ Log Today's Weight"):
        cursor.execute(
            "INSERT INTO weight_logs VALUES (?, ?, ?)",
            (st.session_state.username, new_weight, today)
        )

        conn.commit()

        st.success("Weight logged successfully!")
        st.rerun()


    progress_df = pd.read_sql_query("""
    SELECT date, weight FROM weight_logs
    WHERE username=?
    ORDER BY date
    """, conn, params=(st.session_state.username,))

    if not progress_df.empty:

        # Convert date column to datetime
        progress_df["date"] = pd.to_datetime(progress_df["date"])

        # Show weight trend chart
        st.line_chart(progress_df.set_index("date"))

        # ---------------- RAPID WEIGHT CHANGE DETECTION ----------------
        if len(progress_df) >= 2:

            latest_weight = progress_df.iloc[-1]["weight"]

            week_ago_date = pd.to_datetime(datetime.date.today() - datetime.timedelta(days=7))

            past_week_data = progress_df[progress_df["date"] >= week_ago_date]

            if not past_week_data.empty:
                week_start_weight = past_week_data.iloc[0]["weight"]
                weekly_change = latest_weight - week_start_weight

                st.markdown("### Weight Change Analysis")

                if abs(weekly_change) > 1:
                    st.error(
                        f"Rapid weight change detected: {weekly_change:+.2f} kg in the last 7 days. "
                        "Consider reviewing diet and activity levels."
                    )
                else:
                    st.success(
                        f"Weight change in last 7 days: {weekly_change:+.2f} kg "
                        "(within healthy range)."
                    )

    else:
        st.info("No progress data yet. Start logging your weight.")

    st.markdown("---")


    
    # -------------------------------------------------
    # DAILY CALORIE GUIDANCE
    # -------------------------------------------------
    st.subheader("Daily Calorie Guidance")

    # BMR Calculation
    if gender == "Female":
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5

    activity_factor = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725
    }

    # TRUE Maintenance Calories
    maintenance_calories = bmr * activity_factor[activity]

    # Goal Target Calories
    if goal == "Weight Loss":
        target_calories = maintenance_calories - 300
    elif goal == "Weight Gain":
        target_calories = maintenance_calories + 300
    else:
        target_calories = maintenance_calories

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Maintenance Calories", f"{maintenance_calories:.0f} kcal")

    with col2:
        st.metric("Target Calories", f"{target_calories:.0f} kcal")

    st.caption("Maintenance = calories to keep current weight. Target = adjusted for goal.")


    # -------------------------------------------------
    # REAL WEEKLY WEIGHT PREDICTION (STABLE VERSION)
    # -------------------------------------------------
    st.subheader("Weekly Weight Change Prediction (Based on Your Logs)")

    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

    df_week = pd.read_sql_query("""
    SELECT date, calories FROM food_logs
    WHERE username=? AND date>=?
    """, conn, params=(st.session_state.username, week_ago))


    # -------------------------------------------------
    # STABLE WEEKLY PREDICTION + CONFIDENCE + BEHAVIOR SCORE
    # -------------------------------------------------

    df_week["calories"] = pd.to_numeric(df_week["calories"], errors="coerce")
    daily_totals = df_week.groupby("date")["calories"].sum().reset_index()

    days_logged = len(daily_totals)

    if days_logged < 3:
        st.info("At least 3 logged days are required for reliable prediction.")
    else:
        avg_daily_intake = daily_totals["calories"].mean()
        estimated_weekly_intake = avg_daily_intake * 7
        weekly_required = maintenance_calories * 7

        calorie_difference = estimated_weekly_intake - weekly_required
        predicted_weight_change = calorie_difference / 7700
        predicted_weight = weight + predicted_weight_change

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Avg Daily Intake", f"{avg_daily_intake:.0f} kcal")

        with col2:
            st.metric("Predicted Weekly Change", f"{predicted_weight_change:+.2f} kg")

        # -------------------------------
        # CONFIDENCE SCORE SYSTEM
        # -------------------------------

        logging_consistency = (days_logged / 7) * 100

        calorie_variation = daily_totals["calories"].std()
        if pd.isna(calorie_variation):
            calorie_variation = 0

        stability_score = max(0, 100 - (calorie_variation / 10))

        confidence_score = (logging_consistency * 0.6) + (stability_score * 0.4)
        confidence_score = min(100, max(0, confidence_score))

        st.markdown("### Prediction Confidence Score")
        st.progress(int(confidence_score))

        if confidence_score > 80:
            st.success("High confidence prediction (consistent logging & stable intake).")
        elif confidence_score > 50:
            st.info("Moderate confidence. Logging consistency can improve accuracy.")
        else:
            st.warning("Low confidence. Log daily intake more consistently.")

        # -------------------------------
        # BEHAVIORAL DISCIPLINE SCORE
        # -------------------------------

        difference_from_target = abs(avg_daily_intake - target_calories)

        if difference_from_target < 100:
            discipline_score = 90
        elif difference_from_target < 250:
            discipline_score = 70
        elif difference_from_target < 400:
            discipline_score = 50
        else:
            discipline_score = 30

        st.markdown("### Behavioral Discipline Score")
        st.progress(discipline_score)

        if discipline_score >= 80:
            st.success("Excellent calorie control. Strong discipline.")
        elif discipline_score >= 60:
            st.info("Good control. Minor improvements needed.")
        else:
            st.warning("High deviation from target. Focus on consistency.")

        # -------------------------------
        # Final Weight Trend Insight
        # -------------------------------

        if predicted_weight_change > 0.2:
            st.warning(
                f"If this continues, weight may increase to "
                f"{predicted_weight:.2f} kg next week."
            )
        elif predicted_weight_change < -0.2:
            st.success(
                f"If this continues, weight may decrease to "
                f"{predicted_weight:.2f} kg next week."
            )
        else:
            st.info("Trend indicates weight stability.")

        st.caption("Prediction uses 7700 kcal ≈ 1 kg body weight estimation model.")


    # ---------------- HEALTHY HABITS CARD ----------------
    st.subheader("Daily Compliance Tracker")

    h1, h2, h3 = st.columns(3)

    with h1:
        st.checkbox("Balanced meals")
        st.checkbox("Hydration")

    with h2:
        st.checkbox("Daily activity")
        st.checkbox("7–8 hrs sleep")

    with h3:
        st.checkbox("Portion control")
        st.checkbox("Stress management")

    st.info("Consistency beats perfection. Small steps create big results.")

    # -------------------------------------------------
    # HEALTHY WEIGHT RANGE
    # -------------------------------------------------
    st.subheader("Healthy Weight Range")

    h_m = height / 100
    min_weight = 18.5 * (h_m ** 2)
    max_weight = 24.9 * (h_m ** 2)

    st.info(
        f"For your height ({height} cm), a healthy weight range is "
        f"**{min_weight:.1f} kg – {max_weight:.1f} kg**."
    )

    # -------------------------------------------------
    # ACTIVITY RECOMMENDATIONS
    # -------------------------------------------------
    st.subheader("Activity Recommendations")

    activity_tips = {
        "Sedentary": "Start with 20–30 minutes of walking daily.",
        "Lightly Active": "Include brisk walking, yoga, or cycling.",
        "Moderately Active": "Add strength training 3–4 days per week.",
        "Very Active": "Ensure proper recovery and balanced nutrition."
    }

    st.write(f"**Based on your activity level:** {activity_tips[activity]}")

    # -------------------------------------------------
    # NUTRITION GUIDELINES
    # -------------------------------------------------
    st.subheader("Nutrition Guidelines")

    st.markdown("""
    - Choose **complex carbohydrates** (rice, millets, oats)
    - Include **fiber-rich vegetables** daily
    - Ensure adequate **protein intake**
    - Add **healthy fats** in moderation
    - Stay hydrated (2–3 liters/day)
    """)

    # -------------------------------------------------
    # HEALTHY HABITS CHECKLIST
    # -------------------------------------------------
    st.subheader("Healthy Habits Checklist")

    st.checkbox("Eat regular meals")
    st.checkbox("Avoid sugary drinks")
    st.checkbox("Sleep 7–8 hours daily")
    st.checkbox("Exercise at least 30 minutes")
    st.checkbox("Manage stress effectively")

    st.markdown("---")
    st.info("Healthy weight is a journey — focus on consistency, not perfection.")
# -------------------------------------------------
# COMMON FOODS & CALORIE BURN GUIDE
# -------------------------------------------------
    st.markdown("---")
    st.subheader("Common Foods & Burn Guide")

    food = st.selectbox("Select a food", list(COMMON_FOODS.keys()))

    data = COMMON_FOODS[food]
    st.metric("Calories", f"{data['calories']} kcal")
    st.info(f"Exercise to burn: **{data['exercise']}**")


# =========================================================
# INGREDIENTS GUIDE – FULL NUTRITION INTELLIGENCE SYSTEM
# =========================================================
elif st.session_state.page == "Ingredients Guide":

    st.title("Nutrition Intelligence & Ingredient Guide")
    st.caption("Macro analysis • Health scoring • Smart ingredient education")
    st.markdown("---")

    # =====================================================
    # LOAD USER PROFILE
    # =====================================================
    cursor.execute("""
        SELECT age, gender, height, weight, activity, goal,
               diabetes, acidity, constipation, obesity
        FROM users WHERE username=?
    """, (st.session_state.username,))
    user_data = cursor.fetchone()

    age, gender, height, weight, activity, goal, diabetes, acidity, constipation, obesity = user_data

    # =====================================================
    # SECTION 1 – NUTRITION INTELLIGENCE (CSV BASED)
    # =====================================================
    st.markdown("## Nutrition Intelligence Engine")

    food_list = sorted(calorie_df["category"].unique())
    selected_food = st.selectbox("Select Food to Analyze", food_list)

    food_row = calorie_df[calorie_df["category"] == selected_food]

    if not food_row.empty:

        food = food_row.iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Calories", f"{food['calories_per_100g']} kcal")
        col2.metric("Protein", f"{food['protein_g']} g")
        col3.metric("Fat", f"{food['fat_g']} g")
        col4.metric("Carbs", f"{food['carbs_g']} g")

        st.markdown("---")

             # ---------------- ADVANCED SCORING ENGINE ----------------
        score = 100
        warnings = []
        positives = []

        # Base Calorie Impact
        if food["calories_per_100g"] > 400:
            score -= 30
            warnings.append("Very high calorie density.")
        elif food["calories_per_100g"] > 300:
            score -= 20
            warnings.append("High calorie density.")

        # Sugar Impact (applies to everyone)
        if food["sugar_g"] > 25:
            score -= 25
            warnings.append("Very high sugar content.")
        elif food["sugar_g"] > 10:
            score -= 15
            warnings.append("Moderate sugar content.")

        # Fat Impact
        if food["fat_g"] > 20:
            score -= 20
            warnings.append("High fat content.")

        # Sodium Impact
        if food["sodium_mg"] > 600:
            score -= 20
            warnings.append("High sodium level.")
        elif food["sodium_mg"] > 300:
            score -= 10

        # Fiber Bonus
        if food["fiber_g"] >= 5:
            score += 10
            positives.append("High fiber supports digestion.")

        # Protein Bonus
        if food["protein_g"] >= 15:
            score += 10
            positives.append("High protein content.")

        # Goal-based Adjustments
        if goal == "Weight Loss":
            if food["calories_per_100g"] > 300:
                score -= 15
            if food["fiber_g"] >= 5:
                score += 5

        if goal == "Weight Gain":
            if food["calories_per_100g"] >= 300:
                score += 5

        # Diabetes Adjustments
        if diabetes:
            if food["sugar_g"] > 10:
                score -= 25
                warnings.append("Not ideal for diabetes.")
            if food["glycemic_index"] > 70:
                score -= 20
                warnings.append("High Glycemic Index.")

        # Obesity
        if obesity and food["fat_g"] > 15:
            score -= 15

        # Constipation
        if constipation:
            if food["fiber_g"] < 3:
                score -= 10
                warnings.append("Low fiber for digestion.")
            else:
                score += 5

        # Acidity
        if acidity and selected_food in ["tomato", "fried_food", "chai"]:
            score -= 10
            warnings.append("May trigger acidity.")

        # Clamp Score
        score = max(0, min(100, score))

        st.markdown("### Health Suitability Score")
        st.progress(score)

        if score > 75:
            st.success("Excellent choice based on your profile.")
        elif score > 50:
            st.info("Moderate choice. Watch portion size.")
        else:
            st.warning("Not ideal for your health condition.")

        if warnings:
            st.markdown("### ⚠ Risk Factors")
            for w in warnings:
                st.warning(w)

        if positives:
            st.markdown("### Positive Factors")
            for p in positives:
                st.success(p)

    st.markdown("---")

    # =====================================================
    # SECTION 2 – SMART INGREDIENT EDUCATION LIBRARY
    # =====================================================
    st.markdown("## Smart Ingredient Knowledge Library")

    INGREDIENTS = {
        "Rice": {
            "calories": 130,
            "protein": 2.4,
            "benefits": "Good energy source. Easy to digest.",
            "avoid_if": ["Diabetes (limit portion)"]
        },
        "Broccoli": {
            "calories": 34,
            "protein": 2.8,
            "benefits": "High fiber. Rich in Vitamin C.",
            "avoid_if": []
        },
        "Eggs": {
            "calories": 155,
            "protein": 13,
            "benefits": "High quality protein. Supports muscle growth.",
            "avoid_if": []
        },
        "Chicken Breast": {
            "calories": 165,
            "protein": 31,
            "benefits": "Lean protein source.",
            "avoid_if": []
        },
        "Paneer": {
            "calories": 265,
            "protein": 18,
            "benefits": "High protein & calcium.",
            "avoid_if": ["Obesity (limit portion)"]
        },
        "Banana": {
            "calories": 89,
            "protein": 1.1,
            "benefits": "Quick energy. Rich in potassium.",
            "avoid_if": ["Diabetes (monitor intake)"]
        },
        "Oats": {
            "calories": 389,
            "protein": 16,
            "benefits": "High fiber. Supports heart health.",
            "avoid_if": []
        },
        "Tomato": {
            "calories": 18,
            "protein": 0.9,
            "benefits": "Rich in antioxidants.",
            "avoid_if": ["Acidity (if sensitive)"]
        }
    }

    INGREDIENT_EMOJIS = {
    "Banana": "🍌",
    "Broccoli": "🥦",
    "Chicken Breast": "🍗",
    "Eggs": "🥚",
    "Oats": "🌾",
    "Paneer": "🧀",
    "Rice": "🍚",
    "Tomato": "🍅"
}
    
    search = st.text_input("🔎 Search Ingredient")
    ingredient_list = sorted(INGREDIENTS.keys())

    if search:
        ingredient_list = [
            i for i in ingredient_list
            if search.lower() in i.lower()
        ]

    for item in ingredient_list:
        data = INGREDIENTS[item]

        emoji = INGREDIENT_EMOJIS.get(item, "🥗")
        with st.expander(f"{emoji} {item}"):

            col1, col2 = st.columns(2)
            col1.metric("Calories (per 100g)", f"{data['calories']} kcal")
            col1.metric("Protein (per 100g)", f"{data['protein']} g")

            col2.info(data["benefits"])

            warnings = []

            if diabetes and any("Diabetes" in x for x in data["avoid_if"]):
                warnings.append("⚠ Not ideal for Diabetes.")

            if acidity and any("Acidity" in x for x in data["avoid_if"]):
                warnings.append("⚠ May trigger acidity.")

            if obesity and any("Obesity" in x for x in data["avoid_if"]):
                warnings.append("⚠ High calorie. Portion control recommended.")

            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("Suitable based on your health profile.")

    st.markdown("---")
    st.info("Intelligence engine uses macro-analysis + rule-based health scoring.")


# =========================================================
# WEIGHT LOSS ENGINE – PERSONALIZED VERSION
# =========================================================
elif st.session_state.page == "Lose Weight Safely":

    st.title("Smart Weight Loss Engine")
    st.caption("Personalized • Science-backed • Sustainable fat loss")
    st.markdown("---")

    # ---------------- GET USER DATA ----------------
    cursor.execute("""
        SELECT age, weight, height, gender
        FROM users WHERE username=?
    """, (st.session_state.username,))
    user = cursor.fetchone()

    if user:

        age, weight, height, gender = user

        activity = st.selectbox(
        "Select Your Activity Level",
        ["Sedentary", "Light", "Moderate", "Active", "Very Active"]
)

        # ---------------- BMR CALCULATION ----------------
        if gender == "Male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        # ---------------- ACTIVITY MULTIPLIER ----------------
        ACTIVITY_MULTIPLIERS = {
            "Sedentary": 1.2,
            "Light": 1.375,
            "Moderate": 1.55,
            "Active": 1.725,
            "Very Active": 1.9
        }

        multiplier = ACTIVITY_MULTIPLIERS.get(activity, 1.2)

        maintenance_calories = bmr * multiplier

        # ---------------- SAFE DEFICIT ----------------
        deficit = 500  # 500 kcal/day = ~0.5kg/week
        target_calories = maintenance_calories - deficit

        # ---------------- BMI ----------------
        height_m = height / 100
        bmi = weight / (height_m ** 2)

        # ---------------- DISPLAY METRICS ----------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Your BMI", f"{bmi:.1f}")
        col2.metric("Maintenance Calories", f"{maintenance_calories:.0f} kcal")
        col3.metric("Target Calories", f"{target_calories:.0f} kcal")

        st.markdown("---")

        # ---------------- BMI INTERPRETATION ----------------
        if bmi < 18.5:
            st.info("You are underweight. Focus on healthy weight gain instead.")
        elif 18.5 <= bmi < 25:
            st.success("You are in a healthy weight range.")
        elif 25 <= bmi < 30:
            st.warning("You are overweight. Moderate fat loss recommended.")
        else:
            st.error("Obesity range. Structured fat-loss plan recommended.")

        # ---------------- MACRO BREAKDOWN ----------------
        st.subheader("Suggested Macro Split")

        protein_target = weight * 1.6  # grams
        fat_target = target_calories * 0.25 / 9
        carbs_target = (target_calories - (protein_target * 4 + fat_target * 9)) / 4

        col1, col2, col3 = st.columns(3)

        col1.metric("Protein", f"{protein_target:.0f} g/day")
        col2.metric("Fats", f"{fat_target:.0f} g/day")
        col3.metric("Carbs", f"{carbs_target:.0f} g/day")

        st.markdown("---")

        # ---------------- WEEKLY FAT LOSS ESTIMATE ----------------
        weekly_loss = (deficit * 7) / 7700  # 7700 kcal ≈ 1 kg fat

        st.success(
            f"Expected fat loss: ~{weekly_loss:.2f} kg per week "
            "(if consistent)."
        )

        st.markdown("---")

        # ---------------- PRACTICAL PLAN ----------------
        st.subheader("Action Plan")

        st.markdown("""
        ### Nutrition
        - Prioritize protein in every meal
        - Increase vegetables & fiber
        - Reduce sugar & refined carbs
        - Track portions, not just food types

        ### Activity
        - 8,000–10,000 steps daily
        - Strength training 3–4x/week
        - Cardio 2–3x/week

        ### Lifestyle
        - 7–8 hours sleep
        - Manage stress
        - Drink 2–3 liters water daily
        """)

        st.markdown("---")

        st.warning("""
        ⚠ Avoid:
        - Crash dieting
        - Skipping meals
        - Weight-loss pills
        - Starvation cardio
        """)

        st.info("Sustainable fat loss beats extreme dieting every time.")

    else:
        st.error("User profile incomplete. Please update your details.")
# =========================================================
# 🍭 SUGAR CONTROL – BEHAVIOR ENGINE
# =========================================================
elif st.session_state.page == "Fight Sugar Cravings":

    # ---------------- PAGE HEADER ----------------
    st.markdown("""
    <div class="page-header">
        <h1>Sugar Control Engine</h1>
        <p>Real-time craving intervention & behavior tracking</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- INPUT CARD ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Craving Assessment</h3>", unsafe_allow_html=True)

    craving_level = st.slider(
        "Craving Intensity (1–10)",
        1, 10, 5
    )

    trigger = st.selectbox(
        "Trigger",
        [
            "Stress",
            "Boredom",
            "Hunger",
            "Lack of Sleep",
            "After Meals",
            "Social Event",
            "Habit"
        ]
    )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # ---------------- AI INTERVENTION ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>AI Intervention</h3>", unsafe_allow_html=True)

    if craving_level <= 3:
        intervention = "Hydrate. Wait 10 minutes. Reassess."
    elif craving_level <= 6:
        intervention = "Consume protein (nuts, yogurt, eggs). Stabilize blood sugar."
    else:
        intervention = "10-minute brisk walk + breathing + protein snack."

    if trigger == "Stress":
        intervention += " Add 5-minute deep breathing."
    elif trigger == "Boredom":
        intervention += " Switch environment immediately."
    elif trigger == "Hunger":
        intervention += " Eat a balanced meal instead."
    elif trigger == "Lack of Sleep":
        intervention += " Improve sleep tonight."
    elif trigger == "After Meals":
        intervention += " Brush teeth to reset craving."
    elif trigger == "Social Event":
        intervention += " Choose fruit or small dark chocolate."
    elif trigger == "Habit":
        intervention += " Replace with herbal tea."

    st.success(intervention)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # ---------------- RISK SCORE SYSTEM ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Relapse Risk Score</h3>", unsafe_allow_html=True)

    risk_score = 0

    if craving_level > 7:
        risk_score += 3
    elif craving_level > 5:
        risk_score += 2

    if trigger in ["Stress", "Lack of Sleep"]:
        risk_score += 2
    if trigger == "Habit":
        risk_score += 1

    if risk_score <= 2:
        st.success("Low risk. Maintain control.")
    elif risk_score <= 4:
        st.warning("Moderate risk. Follow intervention strictly.")
    else:
        st.error("High relapse probability. Avoid sugar completely now.")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # ---------------- LOGGING SYSTEM ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Behavior Tracking</h3>", unsafe_allow_html=True)

    if st.button("Log Craving Event", use_container_width=True):

        today = datetime.date.today().isoformat()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sugar_logs (
                username TEXT,
                craving_level INTEGER,
                trigger TEXT,
                date TEXT
            )
        """)

        cursor.execute("""
            INSERT INTO sugar_logs VALUES (?, ?, ?, ?)
        """, (
            st.session_state.username,
            craving_level,
            trigger,
            today
        ))

        conn.commit()
        st.success("Craving logged successfully.")

    # Show weekly craving frequency
    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

    df_week = pd.read_sql_query("""
        SELECT craving_level FROM sugar_logs
        WHERE username=? AND date>=?
    """, conn, params=(st.session_state.username, week_ago))

    if not df_week.empty:
        avg_craving = df_week["craving_level"].mean()
        st.info(f"Weekly Avg Craving Level: {avg_craving:.1f}/10")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    # ---------------- EDUCATION PANEL ----------------
    with st.expander("Proven Sugar Control Strategies"):
        tips = [
            "Hydrate before reacting.",
            "Increase daily protein intake.",
            "Sleep 7–8 hours consistently.",
            "Manage stress proactively.",
            "Avoid grocery shopping hungry.",
            "Use distraction strategy (10-minute rule).",
            "Practice mindful eating."
        ]

        for tip in tips:
            st.write("•", tip)

# =========================================================
# ABOUT PROJECT – UPGRADED
# =========================================================
elif st.session_state.page == "About Project":

    st.title("About the Project")
    st.caption("AI-powered nutrition intelligence for healthier living")

    st.markdown("""
    ### Food Calorie & Fitness Recommendation System

    This project is an **AI-based personalized health and nutrition platform** designed to help users
    make smarter food choices, manage calorie intake, and maintain a healthy lifestyle.

    By combining **Computer Vision, Machine Learning, and Personalized Health Analytics**, the system
    provides real-time food analysis, calorie estimation, and customized fitness guidance.
    """)

    st.markdown("---")

    # ---------------- PROBLEM STATEMENT ----------------
    st.subheader("Problem Statement")
    st.markdown("""
    Many individuals struggle with:
    - Lack of awareness about calorie content in daily meals  
    - Difficulty tracking food intake consistently  
    - Generic diet plans that ignore personal health factors  

    This project addresses these challenges by offering **AI-driven, user-specific insights**
    instead of one-size-fits-all solutions.
    """)

    st.markdown("---")

    # ---------------- SOLUTION OVERVIEW ----------------
    st.subheader("💡 Our Solution")
    st.markdown("""
    The system allows users to:
    - Upload food images for **AI-based food recognition**
    - Instantly estimate **calorie values**
    - Receive **personalized recommendations** based on BMI, activity level, and goals
    - Track weekly food intake and weight progress
    - Learn healthy habits through curated guidance
    """)

    st.markdown("---")

    # ---------------- TECHNOLOGIES ----------------
    st.subheader("Technologies Used")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        **Frontend & UI**
        - Streamlit  
        - Custom CSS  
        - Responsive Layout  
        """)

        st.markdown("""
        **AI & ML**
        - Convolutional Neural Networks (CNN)  
        - TensorFlow / Keras  
        - Image Preprocessing (OpenCV, PIL)  
        """)

    with c2:
        st.markdown("""
        **Backend & Database**
        - Python  
        - SQLite  
        - Secure Password Hashing (SHA-256)  
        """)

        st.markdown("""
        **Data Handling**
        - Pandas  
        - NumPy  
        - CSV-based nutrition dataset  
        """)

    st.markdown("---")

    # ---------------- KEY FEATURES ----------------
    st.subheader("Key Features")
    st.markdown("""
    - Secure login & user-specific data storage  
    - Image-based food calorie detection  
    - Weekly health & calorie reports  
    - BMI-based smart health tips  
    - Weight tracking & prediction  
    - Focus on sustainable and healthy habits  
    """)

    st.markdown("---")

    # ---------------- FUTURE SCOPE ----------------
    st.subheader("Future Enhancements")
    st.markdown("""
    - Multi-food detection from a single image  
    - Portion size estimation  
    - Mobile app integration  
    - Cloud-based deployment  
    - Nutritionist & doctor recommendations  
    """)

    st.markdown("---")

    # ---------------- PROJECT CONTEXT ----------------
    st.subheader("Academic Context")
    st.info("""
    This application is developed as a **Final Year B.Tech Project**, demonstrating the practical
    application of Artificial Intelligence, Machine Learning, and Full-Stack Development
    in the healthcare and fitness domain.
    """)

    st.success("Healthy living starts with informed choices — this project aims to make that easier.")

