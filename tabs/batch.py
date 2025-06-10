import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model_and_tokenizer, predict, DummyPreprocessor
from config import ASPECTS


def batch_input():
    st.header("📂 Phân tích hàng loạt từ file đánh giá")

    model_choice = st.session_state.model_choice
    tokenizer, model = load_model_and_tokenizer(model_choice)
    preprocessor = DummyPreprocessor()

    uploaded_file = st.file_uploader("Tải lên file chứa đánh giá (CSV, Excel, TXT)", type=["csv", "xlsx", "xls", "txt"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".txt"):
                lines = uploaded_file.read().decode("utf-8").splitlines()
                df = pd.DataFrame({'review': lines})
            else:
                st.error("Định dạng không hỗ trợ.")
                return
        except Exception as e:
            st.error(f"Không thể đọc file: {e}")
            return

        if 'review' not in df.columns:
            st.error("File phải có cột tên 'review'")
            return

        st.info(f"Đang phân tích {len(df)} dòng đánh giá...")
        results_all = []

        for i, row in df.iterrows():
            text = str(row['review'])
            output = predict(text, tokenizer, model, aspect_type=model_choice, preprocessor=preprocessor)

            if isinstance(output, str):
                continue

            result, *_ = output
            for aspect, val in result.items():
                results_all.append({
                    'STT': i + 1,
                    'Văn bản': text,
                    'Khía cạnh': aspect,
                    'Cảm xúc': val['label'].capitalize(),
                    'Tự tin': max(val['probs'])
                })

        if not results_all:
            st.warning("Không có khía cạnh nào được phát hiện trong dữ liệu.")
            return

        df_out = pd.DataFrame(results_all)
        st.success(f"Đã phân tích {len(df_out)} dòng kết quả từ {len(df)} review.")

        with st.expander("📄 Xem bảng kết quả"):
            st.dataframe(df_out, use_container_width=True)

        with st.expander("📊 Thống kê cảm xúc"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Tỷ lệ cảm xúc")
                fig1, ax1 = plt.subplots()
                df_out['Cảm xúc'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1, colors=['green', 'red', 'gray'])
                ax1.set_ylabel("")
                st.pyplot(fig1)

            with col2:
                st.markdown("#### Phân bố cảm xúc theo khía cạnh")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df_out, y="Khía cạnh", hue="Cảm xúc", palette='Set2', ax=ax2)
                ax2.set_xlabel("Số lượng")
                st.pyplot(fig2)

        with st.expander("⬇️ Tải kết quả"):
            csv = df_out.copy()
            csv['Tự tin'] = csv['Tự tin'].apply(lambda x: f"{x * 100:.1f}%")
            csv_bytes = csv.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Tải kết quả dạng CSV",
                data=csv_bytes,
                file_name="ket_qua_phan_tich.csv",
                mime='text/csv'
            )

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                csv.to_excel(writer, index=False, sheet_name='PhanTich')
            st.download_button(
                label="📥 Tải kết quả dạng Excel",
                data=excel_buffer.getvalue(),
                file_name="ket_qua_phan_tich.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )