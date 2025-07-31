import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import base64

# --- ฟังก์ชันแปลงภาพเป็น base64 สำหรับใช้ใน CSS ---
def img_to_base64(img_file_path):
    with open(img_file_path, "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()
    return encoded_img

# --- ตั้งค่า ---
st.set_page_config(page_title="Potential CUI Locations", layout="centered")

# --- โหลดรูปภาพและแปลงเป็น base64 ---
logo_path = "Logo.png"
header_path = "Header SCT.png"
logo_base64 = img_to_base64(logo_path)
bg_base64 = img_to_base64(header_path)

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
        font-size: 32px;
        color: #3498db;
        font-weight: bold;
        margin-top: -30px;
        margin-bottom: 16px;
    }}
    .headline {{
        font-size: 28px;
        color: white;
        font-weight: bold;
        margin-bottom: 0px;
        margin-top: 0px;
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
        <span style='color:#fff;font-weight:bold;'>Potential CUI Locations</span>
        </div>
        <div class='subheadline'>Smart CUI Troubleshooting Project</div>
        <div class='desc'>
        Potential CUI Locations can be predicted using AI technology. By leveraging parameters that influence CUI as input, a machine learning model can forecast potential area of CUI. Users simply import the collected data into the model, and the system predicts the likely locations of CUI. This approach enables accurate and efficient assessments, enhancing maintenance planning and prioritization.
        </div>
    </div>
""", unsafe_allow_html=True)

# --- ข้อความแนะนำการอัปโหลด ---
st.markdown(
    "<div style='text-align:center; font-size:20px; color:#3498db; margin-bottom:15px;'>📤 Upload your Excel file to predict CUI severity</div>",
    unsafe_allow_html=True
)

# --- โหลดและฝึกโมเดล (แคชไว้) ---
@st.cache_data
def load_training_model():
    df = pd.read_csv(r"DataTrainingCol1toCol8ForParaX_Col9ForParaY.csv")
    X_train = df.iloc[:, :-1]
    y_train = df.iloc[:, -1]

    encoders = {}
    for col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train.astype(str))

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    return clf, encoders, target_encoder, X_train.columns.tolist()

# โหลดโมเดล
clf, encoders, target_encoder, feature_columns = load_training_model()

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
