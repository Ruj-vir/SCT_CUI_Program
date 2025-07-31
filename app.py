import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import base64
import pickle

# --- ฟังก์ชันแปลงภาพเป็น base64 สำหรับใช้ใน CSS ---
def img_to_base64(img_file_path):
    with open(img_file_path, "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()
    return encoded_img

# --- ตั้งค่า ---
st.set_page_config(page_title="Water Enter Point Detector", layout="centered")

# --- โหลดรูปภาพและแปลงเป็น base64 ---
logo_path = "Logo.png"
header_path = "Header SCT.png"
logo_base64 = img_to_base64(logo_path)
bg_base64 = img_to_base64(header_path)

# --- CSS และส่วนหัวเว็บ ---
st.markdown(f"""
    <style>
    .banner {{
        background-image: url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        padding: 40px 40px 20px 40px;
        border-radius: 0px 0px 20px 20px;
        margin-bottom: 30px;
    }}
    .logo {{
        margin-bottom: 10px;
    }}
    .headline {{
        font-size: 28px;
        color: white;
        font-weight: bold;
        margin-bottom: 0px;
    }}
    .subheadline {{
        font-size: 20px;
        color: white;
        margin-bottom: 18px;
        margin-top: 5px;
        font-weight: 300;
    }}
    .desc {{
        color: #d0e6fa;
        font-size: 15px;
        margin-top: 0;
    }}
    </style>

    <div class='banner'>
        <div class='logo'>
            <img src="data:image/png;base64,{logo_base64}" height="48">
        </div>
        <div class='headline'>
            <span style='color:#222;font-weight:bold;'>POTENTIAL&nbsp;|&nbsp;</span>
            <span style='color:#fff;'>WATER ENTER POINT</span>
        </div>
        <div class='subheadline'>Smart CUI Troubleshooting Project</div>
        <div class='desc'>
            Using this artificial intelligence model, you can efficiently detect potential water enter point from images captured during inspections.
            It is built with YoloV8 components and utilities, requiring minimal modification for your specific use case. Simply import the images into the CIRA CORE platform,
            and the model will analyze them to identify potential areas of water enter point.
        </div>
    </div>
""", unsafe_allow_html=True)

# --- ข้อความแนะนำการอัปโหลด ---
st.markdown(
    "<div style='text-align:center; font-size:20px; color:#3498db; margin-bottom:15px;'>📤 Upload your Excel file to predict potential water enter point.</div>",
    unsafe_allow_html=True
)

# --- โหลดโมเดลจาก .pkl ---
@st.cache_resource
def load_model_from_pickle():
    with open("model_and_encoders.pkl", "rb") as f:
        clf, encoders, target_encoder, feature_columns = pickle.load(f)
    return clf, encoders, target_encoder, feature_columns

clf, encoders, target_encoder, feature_columns = load_model_from_pickle()

# --- อัปโหลดและทำนาย Excel ---
uploaded_file = st.file_uploader("Upload Excel (.xlsx) ที่มี Row 2 เป็นข้อมูล", type=["xlsx"])

if uploaded_file is not None:
    try:
        manual_input = pd.read_excel(uploaded_file, header=0)
        st.write("📄 ข้อมูลที่อัปโหลด:")
        st.dataframe(manual_input)

        manual_input = manual_input[feature_columns]
        unknown_warning = []

        for col in manual_input.columns:
            if col in encoders:
                test_values = manual_input[col].astype(str)
                known_values = set(encoders[col].classes_)
                unknown_values = set(test_values) - known_values

                if unknown_values:
                    unknown_warning.append(f"❌ ค่าที่ไม่รู้จักในคอลัมน์ '{col}': {unknown_values}")
                else:
                    manual_input[col] = encoders[col].transform(test_values)

        if unknown_warning:
            for warn in unknown_warning:
                st.warning(warn)
            st.stop()

        # ทำนายผล
        y_pred = clf.predict(manual_input)
        predicted_label = target_encoder.inverse_transform(y_pred)

        # แสดงผลทำนายด้วยสีตามค่า
        if predicted_label[0] == "No":
            st.error(f"❌ ผลทำนายคือ: **{predicted_label[0]}**")
        else:
            st.success(f"✅ ผลทำนายคือ: **{predicted_label[0]}**")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์: {e}")
