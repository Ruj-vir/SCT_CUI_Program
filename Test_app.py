# เขียนเว็บ Streamlit

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Upload File Demo", layout="centered")

st.title("📂 Upload File Viewer")

# ให้ผู้ใช้เลือกไฟล์
uploaded_file = st.file_uploader("อัปโหลดไฟล์ Excel (.xlsx) หรือ CSV", type=["xlsx", "csv"])

# ตรวจสอบว่ามีไฟล์อัปโหลดหรือไม่
if uploaded_file is not None:
    # แสดงชื่อไฟล์
    st.success(f"ไฟล์ที่อัปโหลด: {uploaded_file.name}")

    # ตรวจสอบประเภทไฟล์
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("📝 ข้อมูลในไฟล์:")
        st.dataframe(df)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")