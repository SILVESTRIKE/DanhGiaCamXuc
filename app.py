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

# Cài đặt ngưỡng tự tin với thanh trượt và nút reset trong cùng hàng
DEFAULT_CONFIDENCE = 0.7

# Xử lý reset trước khi tạo widget
if "confidence_threshold" not in st.session_state:
    st.session_state.confidence_threshold = DEFAULT_CONFIDENCE

with st.sidebar.container():
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("### ")
        if st.button("🔁", help="Đặt lại về mặc định"):
            # Gán lại session_state và rerun, NHƯNG phải làm trước khi tạo slider
            st.session_state.confidence_threshold = DEFAULT_CONFIDENCE

    # Sau khi xử lý reset, mới được render slider
    with col1:
        st.slider(
            "Ngưỡng tự tin",
            min_value=0.3,
            max_value=1.0,
            step=0.1,
            key="confidence_threshold",
        )

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