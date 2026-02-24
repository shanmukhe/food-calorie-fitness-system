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

/* =====================================================
   GLOBAL – CLEAN DARK BASE
===================================================== */
body {
    background-color: #0F172A;
    color: #E2E8F0;
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 2rem;
}

/* =====================================================
   MAIN PAGE – NO OUTER GRADIENT
===================================================== */
.main {
    background-color: #0F172A;
}

/* =====================================================
   CARD SYSTEM – SOFT DARK GRADIENT
===================================================== */
.card {
    background: linear-gradient(145deg, #1E293B, #172033);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.metric-card {
    background: linear-gradient(145deg, #1E293B, #141c2b);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.04);
    text-align: left;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.metric-label {
    font-size: 13px;
    color: #94A3B8;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 22px;
    font-weight: 600;
    color: #F8FAFC;
}

/* =====================================================
   BUTTONS
===================================================== */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #4F46E5);
    border-radius: 12px;
    border: none;
    color: white;
    font-weight: 600;
    padding: 0.65rem 1rem;
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(99,102,241,0.4);
}

/* =====================================================
   INPUTS
===================================================== */
div[data-baseweb="input"] > div {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

/* =====================================================
   SIDEBAR – VISUALLY IMPACTFUL
===================================================== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220, #111827);
    border-right: 1px solid rgba(255,255,255,0.08);
    box-shadow: inset -3px 0 20px rgba(0,0,0,0.4);
}

/* Brand Card */
.brand-card {
    background: linear-gradient(145deg, #312E81, #1E1B4B);
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.brand-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 4px;
}

.brand-sub {
    font-size: 12px;
    opacity: 0.8;
    margin-bottom: 10px;
}

/* NAV LABEL */
.nav-label {
    font-size: 11px;
    letter-spacing: 1px;
    color: #9CA3AF;
    margin-bottom: 10px;
}

/* RADIO NAV STYLE */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: transparent;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 6px;
    transition: all 0.2s ease-in-out;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(99,102,241,0.15);
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, #6366F1, #4F46E5);
    color: white;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4);
}

/* Logout Button */
.logout-btn button {
    background: linear-gradient(135deg, #EF4444, #DC2626) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

.logout-btn button:hover {
    box-shadow: 0 8px 20px rgba(239,68,68,0.4);
}

/* =====================================================
   SECTION TITLES
===================================================== */
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 12px;
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
# SIDEBAR – CLEAN MODERN STRUCTURE
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
    "About"
]

if st.session_state.page not in menu_options:
    st.session_state.page = "Home"

# ---------- BRAND CARD ----------
st.sidebar.markdown(f"""
<div class="brand-card">
    <div class="brand-title">Food Fitness</div>
    <div class="brand-sub">AI Nutrition Platform</div>
    <hr style="border-color:#334155;">
    <small style="color:#94A3B8;">Welcome</small><br>
    <b>{st.session_state.username}</b>
</div>
""", unsafe_allow_html=True)

# ---------- NAVIGATION ----------
st.sidebar.markdown('<div class="nav-label">NAVIGATION</div>', unsafe_allow_html=True)

selected_page = st.sidebar.radio(
    "",
    menu_options,
    index=menu_options.index(st.session_state.page)
)

st.session_state.page = selected_page

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# ---------- LOGOUT ----------
st.sidebar.markdown('<div class="logout-btn">', unsafe_allow_html=True)
logout = st.sidebar.button("Logout", use_container_width=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

if logout:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.page = "Home"
    st.rerun()

# =========================================================
# HOME – CLEAN DASHBOARD
# =========================================================
if st.session_state.page == "Home":

    st.title("Dashboard")
    st.caption("Your personalized health overview")

    # ---------------- LOAD PROFILE ----------------
    cursor.execute("""
        SELECT age, gender, height, weight, activity, goal
        FROM users WHERE username=?
    """, (st.session_state.username,))
    profile = cursor.fetchone()

    if profile and all(profile):

        age, gender, height, weight, activity, goal = profile
        bmi = bmi_calc(weight, height)

        # ---------------- SNAPSHOT CARD ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Health Snapshot")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Weight", f"{weight} kg")

        with c2:
            metric_card("BMI", f"{bmi:.2f}")

        with c3:
            metric_card("Activity", activity)

        with c4:
            metric_card("Goal", goal)

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- INSIGHT CARD ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("AI Insight")

        if bmi < 18.5:
            st.info("Underweight. Gradual calorie surplus + strength training recommended.")
        elif bmi < 25:
            st.success("Healthy BMI range. Maintain consistency.")
        elif bmi < 30:
            st.warning("Slightly above ideal range. Moderate calorie control advised.")
        else:
            st.error("High BMI detected. Structured fat-loss strategy recommended.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- QUICK ACTIONS ----------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Quick Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Analyze Food", use_container_width=True):
                st.session_state.page = "Analyze Food"
                st.rerun()

        with col2:
            if st.button("View Health Insights", use_container_width=True):
                st.session_state.page = "Health Insights"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Complete your profile to activate dashboard insights.")

    # ---------------- DAILY TIP ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Today’s Health Tip")

    tips = [
        "Drink water before meals to reduce overeating.",
        "Consistency beats intensity.",
        "Half your plate should be vegetables.",
        "Daily walking improves metabolic health.",
        "Sleep quality affects fat loss."
    ]

    st.info(np.random.choice(tips))

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# ANALYZE FOOD – CLEAN STRUCTURED UI
# =========================================================
elif st.session_state.page == "Analyze Food":

    st.title("AI Food Analysis")
    st.caption("Upload a food image or log manually")

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
        "Select Mode",
        ["Upload Image", "Manual Entry"],
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
                    if st.button("Analyze", use_container_width=True):
                        with st.spinner("Running AI analysis..."):
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

            st.markdown("### Top Predictions")

            for food, conf in st.session_state.top_results:
                st.write(f"**{food}** — {conf:.2f}% confidence")

            st.session_state.selected_food = st.selectbox(
                "Confirm Food",
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
                col2.metric("Calories", f"{total_calories:.0f} kcal")

                if st.button("Confirm & Log", use_container_width=True):

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

        manual_food = st.selectbox(
            "Select Food",
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
            col2.metric("Calories", f"{total_calories:.0f} kcal")

            if st.button("Log Food", use_container_width=True):

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

                st.success("Food logged successfully.")

    st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# HEALTH INSIGHTS – CLEAN STRUCTURED VERSION
# =========================================================
elif st.session_state.page == "Health Insights":

    st.title("Health Insights")
    st.caption("Track intake, burn calories, and monitor trends")

    # ---------------- LOAD PROFILE ----------------
    cursor.execute("""
        SELECT age, gender, height, weight, activity, goal
        FROM users WHERE username=?
    """, (st.session_state.username,))
    profile = cursor.fetchone()

    if not profile:
        st.warning("Complete your profile first.")
        st.stop()

    age, gender, height, profile_weight, activity, goal = profile
    weight = get_latest_weight(st.session_state.username, profile_weight)

    maintenance_calories, target_calories = calculate_target_calories(
        age, gender, height, weight, activity, goal
    )

    today = datetime.date.today().isoformat()

    # ==================================================
    # TODAY SUMMARY
    # ==================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Today’s Summary")

    df_today = pd.read_sql_query("""
        SELECT calories FROM food_logs
        WHERE username=? AND date=?
    """, conn, params=(st.session_state.username, today))

    df_ex_today = pd.read_sql_query("""
        SELECT calories_burned FROM exercise_logs
        WHERE username=? AND date=?
    """, conn, params=(st.session_state.username, today))

    consumed_today = df_today["calories"].sum() if not df_today.empty else 0
    burned_today = df_ex_today["calories_burned"].sum() if not df_ex_today.empty else 0
    net_calories = consumed_today - burned_today

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Target", f"{target_calories:.0f} kcal")

    with c2:
        metric_card("Consumed", f"{consumed_today:.0f} kcal")

    with c3:
        metric_card("Burned", f"{burned_today:.0f} kcal")

    with c4:
        metric_card("Net", f"{net_calories:.0f} kcal")

    if net_calories > target_calories:
        st.error("Calorie surplus today.")
    else:
        st.success("Within calorie target.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ==================================================
    # EXERCISE LOG
    # ==================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Log Exercise")

    exercise_type = st.selectbox(
        "Exercise Type",
        ["Walking", "Jogging", "Running", "Cycling", "Yoga"]
    )

    minutes = st.number_input("Duration (minutes)", 0, 300, step=5)

    MET_VALUES = {
        "Walking": 3.5,
        "Jogging": 7,
        "Running": 11,
        "Cycling": 8,
        "Yoga": 3
    }

    if st.button("Log Exercise", use_container_width=True):
        if minutes == 0:
            st.warning("Enter duration.")
        else:
            met = MET_VALUES[exercise_type]
            calories_burned = met * weight * (minutes / 60)

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
            st.success(f"{calories_burned:.0f} kcal burned logged.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ==================================================
    # WEEKLY ENERGY BALANCE
    # ==================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Weekly Energy Balance")

    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

    df_food_week = pd.read_sql_query("""
        SELECT date, calories FROM food_logs
        WHERE username=? AND date>=?
    """, conn, params=(st.session_state.username, week_ago))

    df_ex_week = pd.read_sql_query("""
        SELECT date, calories_burned FROM exercise_logs
        WHERE username=? AND date>=?
    """, conn, params=(st.session_state.username, week_ago))

    food_grouped = df_food_week.groupby("date")["calories"].sum().reset_index()
    ex_grouped = df_ex_week.groupby("date")["calories_burned"].sum().reset_index()

    merged = pd.merge(food_grouped, ex_grouped, on="date", how="outer").fillna(0)
    merged["net"] = merged["calories"] - merged["calories_burned"]

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=merged["date"],
        y=merged["calories"],
        name="Consumed"
    ))

    fig.add_trace(go.Bar(
        x=merged["date"],
        y=merged["calories_burned"],
        name="Burned"
    ))

    fig.add_trace(go.Scatter(
        x=merged["date"],
        y=merged["net"],
        name="Net",
        mode="lines+markers"
    ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        xaxis_title="Date",
        yaxis_title="Calories"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ==================================================
    # AI INSIGHT
    # ==================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI Weekly Insight")

    if not merged.empty:
        weekly_net_avg = merged["net"].mean()
        diff = weekly_net_avg - target_calories

        if diff > 150:
            st.error("Consistent calorie surplus. Weight gain likely.")
        elif diff < -150:
            st.success("Consistent deficit. Fat loss expected.")
        else:
            st.info("Near maintenance. Weight stability expected.")

        predicted_change = diff * 7 / 7700

        st.metric("Predicted Weekly Change",
                  f"{predicted_change:+.2f} kg")
    else:
        st.info("Not enough weekly data.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ==================================================
    # ADAPTIVE TARGET
    # ==================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Adaptive Calorie Target")

    if not merged.empty:
        weekly_net_avg = merged["net"].mean()
        adaptive_target = target_calories

        if goal == "Weight Loss" and weekly_net_avg > target_calories + 150:
            adaptive_target -= 150
        elif goal == "Weight Gain" and weekly_net_avg < target_calories - 150:
            adaptive_target += 150

        st.metric("Recommended Target",
                  f"{adaptive_target:.0f} kcal/day")
    else:
        st.info("Requires more logged data.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FACTS & MYTHS – CLEAN EDUCATION MODULE
# =========================================================
elif st.session_state.page == "Facts & Myths":

    st.title("Food & Calories – Myths vs Facts")
    st.caption("Evidence-based clarity for smarter decisions")

    # =====================================================
    # MYTHS SECTION
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Common Nutrition Myths")

    myths_facts = [
        ("Eating less always guarantees weight loss",
         "Extreme restriction slows metabolism. Sustainable balance matters."),

        ("All calories are the same",
         "Nutrient quality affects hormones, hunger, and energy."),

        ("Skipping meals burns fat faster",
         "Often leads to overeating and unstable blood sugar."),

        ("Carbs make you fat",
         "Complex carbs are essential energy sources."),

        ("Fat-free foods are always healthy",
         "Many contain added sugar. Healthy fats are necessary."),

        ("Late-night eating causes weight gain",
         "Total daily intake matters more than timing."),

        ("Fruits can be eaten without limits",
         "Fruits are healthy but still contain natural sugars."),

        ("Exercise alone ensures fat loss",
         "Nutrition plays the larger role."),

        ("Water burns fat instantly",
         "Hydration supports health but doesn’t directly burn fat."),

        ("You must avoid favorite foods completely",
         "Balance and portion control are more sustainable."
        )
    ]

    col1, col2 = st.columns(2)

    for i, (myth, fact) in enumerate(myths_facts):
        container = col1 if i % 2 == 0 else col2
        with container:
            st.markdown(f"**Myth:** {myth}")
            st.caption(f"Fact: {fact}")
            st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # FUNDAMENTALS
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Nutrition Fundamentals")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        metric_card("Protein", "Muscle repair")

    with m2:
        metric_card("Fiber", "Hunger control")

    with m3:
        metric_card("Healthy Fats", "Hormone balance")

    with m4:
        metric_card("Low Sugar", "Metabolic health")

    st.info("Consistency beats restriction.")
    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # QUIZ MODULE
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Myth or Fact Quiz")

    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0

    if "quiz_question" not in st.session_state:
        st.session_state.quiz_question = None

    if "quiz_answer" not in st.session_state:
        st.session_state.quiz_answer = None

    if "quiz_feedback" not in st.session_state:
        st.session_state.quiz_feedback = None

    quiz_pool = [
        ("Eating less always guarantees weight loss", "Myth"),
        ("Total calorie intake matters more than timing", "Fact"),
        ("Carbs automatically cause fat gain", "Myth"),
        ("Protein supports muscle repair", "Fact"),
        ("Drinking water burns fat directly", "Myth"),
        ("Fiber improves digestion", "Fact"),
        ("Exercise alone causes fat loss", "Myth"),
        ("Healthy fats support hormones", "Fact"),
    ]

    import random

    if st.button("New Question") or st.session_state.quiz_question is None:
        question, answer = random.choice(quiz_pool)
        st.session_state.quiz_question = question
        st.session_state.quiz_answer = answer
        st.session_state.quiz_feedback = None

    if st.session_state.quiz_question:

        st.write(f"**Question:** {st.session_state.quiz_question}")

        user_choice = st.radio(
            "Your Answer",
            ["Myth", "Fact"],
            horizontal=True
        )

        if st.button("Submit"):
            if user_choice == st.session_state.quiz_answer:
                st.session_state.quiz_score += 1
                st.session_state.quiz_feedback = "correct"
            else:
                st.session_state.quiz_feedback = "wrong"

        if st.session_state.quiz_feedback == "correct":
            st.success("Correct!")

        elif st.session_state.quiz_feedback == "wrong":
            st.error(f"Incorrect. Answer: {st.session_state.quiz_answer}")

        st.metric("Score", st.session_state.quiz_score)

        if st.button("Reset Quiz"):
            st.session_state.quiz_score = 0
            st.session_state.quiz_question = None
            st.session_state.quiz_feedback = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# HEALTHY WEIGHT TOOLKIT – CLEAN STRUCTURED VERSION
# =========================================================
elif st.session_state.page == "Healthy Weight Toolkit":

    st.title("Healthy Weight Toolkit")
    st.caption("Personal metrics • Progress • Predictive intelligence")

    # ---------------- LOAD PROFILE ----------------
    cursor.execute("""
        SELECT age, gender, height, weight, activity, goal
        FROM users WHERE username=?
    """, (st.session_state.username,))
    profile = cursor.fetchone()

    if not profile:
        st.warning("Complete your profile first.")
        st.stop()

    age, gender, height, profile_weight, activity, goal = profile
    weight = get_latest_weight(st.session_state.username, profile_weight)

    bmi = bmi_calc(weight, height)

    # =====================================================
    # BODY METRICS
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Body Metrics")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("BMI", f"{bmi:.2f}")

    with c2:
        metric_card("Weight", f"{weight} kg")

    with c3:
        metric_card("Goal", goal)

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # WEIGHT PROGRESS
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Weight Progress")

    today = datetime.date.today().isoformat()

    new_weight = st.number_input(
        "Log Today's Weight (kg)",
        30.0, 200.0,
        value=float(weight),
        step=0.1
    )

    if st.button("Log Weight", use_container_width=True):
        cursor.execute(
            "INSERT INTO weight_logs VALUES (?, ?, ?)",
            (st.session_state.username, new_weight, today)
        )
        conn.commit()
        st.success("Weight logged.")
        st.rerun()

    progress_df = pd.read_sql_query("""
        SELECT date, weight FROM weight_logs
        WHERE username=?
        ORDER BY date
    """, conn, params=(st.session_state.username,))

    if not progress_df.empty:
        progress_df["date"] = pd.to_datetime(progress_df["date"])
        st.line_chart(progress_df.set_index("date"))
    else:
        st.info("No weight history yet.")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # CALORIE GUIDANCE
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Calorie Guidance")

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

    maintenance = bmr * activity_factor[activity]

    if goal == "Weight Loss":
        target = maintenance - 300
    elif goal == "Weight Gain":
        target = maintenance + 300
    else:
        target = maintenance

    col1, col2 = st.columns(2)

    with col1:
        metric_card("Maintenance", f"{maintenance:.0f} kcal")

    with col2:
        metric_card("Target", f"{target:.0f} kcal")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # WEEKLY PREDICTION
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Weekly Prediction")

    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

    df_week = pd.read_sql_query("""
        SELECT date, calories FROM food_logs
        WHERE username=? AND date>=?
    """, conn, params=(st.session_state.username, week_ago))

    df_week["calories"] = pd.to_numeric(df_week["calories"], errors="coerce")
    daily_totals = df_week.groupby("date")["calories"].sum().reset_index()

    if len(daily_totals) < 3:
        st.info("Log at least 3 days of food for prediction.")
    else:
        avg_intake = daily_totals["calories"].mean()
        weekly_diff = (avg_intake * 7) - (maintenance * 7)
        predicted_change = weekly_diff / 7700

        col1, col2 = st.columns(2)

        with col1:
            metric_card("Avg Intake", f"{avg_intake:.0f} kcal")

        with col2:
            metric_card("Predicted Change", f"{predicted_change:+.2f} kg")

        if predicted_change > 0.2:
            st.warning("Weight gain trend detected.")
        elif predicted_change < -0.2:
            st.success("Weight loss trend detected.")
        else:
            st.info("Weight stability likely.")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # HABIT TRACKER
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Daily Habit Tracker")

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("Balanced meals")
        st.checkbox("Hydration")

    with col2:
        st.checkbox("Daily activity")
        st.checkbox("7–8 hrs sleep")

    st.info("Consistency compounds over time.")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # HEALTH RANGE
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Healthy Weight Range")

    h_m = height / 100
    min_w = 18.5 * (h_m ** 2)
    max_w = 24.9 * (h_m ** 2)

    st.info(f"Ideal range for your height: {min_w:.1f} kg – {max_w:.1f} kg")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# SUGAR CONTROL – CLEAN BEHAVIOR ENGINE
# =========================================================
elif st.session_state.page == "Fight Sugar Cravings":

    st.title("Sugar Control Engine")
    st.caption("Craving intervention • Risk scoring • Weekly behavior tracking")

    # =====================================================
    # CRAVING INPUT
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Craving Assessment")

    craving_level = st.slider("Craving Intensity (1–10)", 1, 10, 5)

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

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # AI INTERVENTION
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI Intervention")

    if craving_level <= 3:
        intervention = "Hydrate and wait 10 minutes."
    elif craving_level <= 6:
        intervention = "Have protein (nuts, yogurt, eggs)."
    else:
        intervention = "Take a 10-minute brisk walk + protein snack."

    trigger_actions = {
        "Stress": "Add 5-minute deep breathing.",
        "Boredom": "Change environment immediately.",
        "Hunger": "Eat a balanced meal instead.",
        "Lack of Sleep": "Prioritize sleep tonight.",
        "After Meals": "Brush teeth to reset craving.",
        "Social Event": "Choose fruit or dark chocolate.",
        "Habit": "Replace with herbal tea."
    }

    intervention += " " + trigger_actions.get(trigger, "")

    st.success(intervention)
    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # RISK SCORE
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Relapse Risk Score")

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
        st.success("Low relapse risk.")
    elif risk_score <= 4:
        st.warning("Moderate risk. Follow intervention strictly.")
    else:
        st.error("High relapse probability. Avoid sugar completely.")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # TRACKING SYSTEM
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Behavior Tracking")

    today = datetime.date.today().isoformat()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sugar_logs (
            username TEXT,
            craving_level INTEGER,
            trigger TEXT,
            date TEXT
        )
    """)

    if st.button("Log Craving Event", use_container_width=True):
        cursor.execute("""
            INSERT INTO sugar_logs VALUES (?, ?, ?, ?)
        """, (
            st.session_state.username,
            craving_level,
            trigger,
            today
        ))
        conn.commit()
        st.success("Craving logged.")
        st.rerun()

    week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

    df_week = pd.read_sql_query("""
        SELECT craving_level FROM sugar_logs
        WHERE username=? AND date>=?
    """, conn, params=(st.session_state.username, week_ago))

    if not df_week.empty:
        avg_craving = df_week["craving_level"].mean()
        metric_card("Weekly Avg Craving", f"{avg_craving:.1f}/10")
    else:
        st.info("No weekly data yet.")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # EDUCATION PANEL
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Proven Sugar Control Strategies")

    tips = [
        "Hydrate before reacting.",
        "Increase daily protein intake.",
        "Sleep 7–8 hours consistently.",
        "Manage stress proactively.",
        "Avoid grocery shopping hungry.",
        "Use the 10-minute delay rule.",
        "Practice mindful eating."
    ]

    for tip in tips:
        st.write("•", tip)

    st.markdown("</div>", unsafe_allow_html=True)
# =========================================================
# ABOUT PROJECT – CLEAN PRODUCT OVERVIEW
# =========================================================
elif st.session_state.page == "About Project":

    st.title("About the Project")
    st.caption("AI-powered nutrition intelligence platform")

    # =====================================================
    # OVERVIEW
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Project Overview")

    st.write("""
    The Food Calorie & Fitness Recommendation System is an AI-based 
    personalized health platform designed to help users make smarter 
    nutrition decisions and maintain a sustainable lifestyle.
    """)

    st.write("""
    It combines computer vision, machine learning, and behavioral 
    analytics to deliver real-time food recognition, calorie estimation,
    and adaptive health recommendations.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # PROBLEM & SOLUTION
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Problem & Solution")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Challenges**")
        st.write("- Poor awareness of calorie intake")
        st.write("- Inconsistent food tracking")
        st.write("- Generic diet advice")
        st.write("- Lack of personalized insights")

    with col2:
        st.markdown("**Our Approach**")
        st.write("- AI-based food recognition")
        st.write("- Personalized calorie targets")
        st.write("- Weekly behavior analytics")
        st.write("- Health-condition aware guidance")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # TECHNOLOGY STACK
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Technology Stack")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Frontend**")
        st.write("- Streamlit")
        st.write("- Custom CSS")

        st.markdown("**AI & ML**")
        st.write("- CNN (TensorFlow / Keras)")
        st.write("- Image preprocessing (PIL / OpenCV)")

    with c2:
        st.markdown("**Backend**")
        st.write("- Python")
        st.write("- SQLite")

        st.markdown("**Data Processing**")
        st.write("- Pandas")
        st.write("- NumPy")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # KEY FEATURES
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Key Features")

    st.write("- Secure user authentication")
    st.write("- Image-based food calorie detection")
    st.write("- Weekly calorie & energy analysis")
    st.write("- Weight tracking & prediction")
    st.write("- Sugar craving intervention engine")
    st.write("- Adaptive calorie target system")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # FUTURE ROADMAP
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Future Enhancements")

    st.write("- Multi-food detection in single image")
    st.write("- Portion size estimation via AI")
    st.write("- Mobile app integration")
    st.write("- Cloud deployment")
    st.write("- Professional nutritionist integration")

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # CONTEXT
    # =====================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Project Context")

    st.info("""
    Developed as a Final Year B.Tech project demonstrating
    practical implementation of Artificial Intelligence,
    Machine Learning, and Full-Stack Development in healthcare.
    """)

    st.markdown("</div>", unsafe_allow_html=True)