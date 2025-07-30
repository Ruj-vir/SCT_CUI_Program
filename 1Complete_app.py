import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Predictor with Upload", layout="centered")
st.title("📊 ทำนายผลด้วย Random Forest")

# ---------- 1. โหลดชุดฝึก Training CSV ----------
@st.cache_data
def load_training_model():
    df = pd.read_csv(r"C:\Users\26009623\Downloads\CUi_Project_\N-Ruj\Program_Ruj_StreamLit\DataSet_ruj\DataTrainingCol1toCol8ForParaX_Col9ForParaY.csv")
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

# โหลดโมเดลและ encoder
clf, encoders, target_encoder, feature_columns = load_training_model()

# ---------- 2. Upload Excel แล้ว Predict ----------
uploaded_file = st.file_uploader("📤 อัปโหลดไฟล์ Excel (.xlsx) ที่มี Row 2 เป็นข้อมูล", type=["xlsx"])

if uploaded_file is not None:
    try:
        manual_input = pd.read_excel(uploaded_file, header=0)

        st.write("📝 ข้อมูลที่อัปโหลด:")
        st.dataframe(manual_input)

        # จัดเรียงคอลัมน์ให้ตรงกับโมเดล
        manual_input = manual_input[feature_columns]

        unknown_warning = []

        # แปลงข้อมูลด้วย encoder ที่ฝึกไว้
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

        # ---------- ทำนายผล ----------
        y_pred = clf.predict(manual_input)
        predicted_label = target_encoder.inverse_transform(y_pred)

        st.success(f"✅ ผลทำนายคือ: **{predicted_label[0]}**")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์: {e}")
