import streamlit as st
from tabs.input import input
from tabs.chart import chart
from tabs.history import history
from tabs.stats import stats
from tabs.batch import batch_input

st.set_page_config(page_title="Phân tích cảm xúc đa mô hình", layout="wide")

st.sidebar.title("🧠 Phân tích cảm xúc")
model_choice = st.sidebar.selectbox("Chọn mô hình", ["Nhà hàng", "Khách sạn"])
page = st.sidebar.radio("Chọn trang", ["Nhập liệu", "Biểu đồ", "Lịch sử", "Thống kê", "Phân tích từ file"])


st.session_state.model_choice = model_choice

if page == "Nhập liệu":
    input()
elif page == "Biểu đồ":
    chart()
elif page == "Lịch sử":
    history()
elif page == "Thống kê":
    stats()
elif page == "Phân tích từ file":
    batch_input()