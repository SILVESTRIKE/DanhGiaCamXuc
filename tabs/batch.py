import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model_and_tokenizer, predict, DummyPreprocessor
from config import ASPECTS
import xlrd
def batch_input():
    st.header("📂 Phân tích hàng loạt từ file đánh giá")

    model_choice = st.session_state.model_choice
    tokenizer, model = load_model_and_tokenizer(model_choice)
    preprocessor = DummyPreprocessor()

    uploaded_file = st.file_uploader(
        "Tải lên file chứa đánh giá (Excel hoặc TXT)", 
        type=["xlsx", "xls", "txt"]
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".txt"):
                lines = uploaded_file.read().decode("utf-8").splitlines()
                df = pd.DataFrame({'review': lines})
            else:
                st.error("Định dạng không hỗ trợ. Chỉ chấp nhận Excel (.xlsx, .xls) hoặc text (.txt).")
                return
        except Exception as e:
            st.error(f"Không thể đọc file: {e}")
            return

        if 'review' not in df.columns:
            st.error("File phải có cột tên 'review'")
            return

        results_all = []
        status_placeholder = st.empty()

        with status_placeholder.container():
            st.write(f"Đang phân tích {len(df)} dòng đánh giá...")
            progress_bar = st.progress(0)

        for idx, row in enumerate(df.itertuples(index=False)):
            try:
                text = str(row.review)
                if not text.strip():
                    continue
            except AttributeError:
                status_placeholder.error("Lỗi đọc cột 'review'. Vui lòng kiểm tra lại tên cột trong file.")
                st.stop()

            output = predict(text, tokenizer, model, aspect_type=model_choice, preprocessor=preprocessor,confidence_threshold=st.session_state.confidence_threshold)

            if isinstance(output, str) or not isinstance(output, (list, tuple)) or len(output) == 0:
                continue

            result, *_ = output
            if isinstance(result, str):
                continue

            for aspect, val in result.items():
                results_all.append({
                    'STT': str(idx + 1),
                    'Văn bản': text,
                    'Khía cạnh': aspect,
                    'Cảm xúc': val['label'].capitalize(),
                    'Tự tin': max(val['probs'])
                })

            progress_bar.progress((idx + 1) / len(df))

        df_out = pd.DataFrame(results_all)
        status_placeholder.success(f"Đã phân tích xong! Tìm thấy {len(df_out)} khía cạnh từ {len(df)} review.")
                
        if not results_all:
            st.warning("Không có khía cạnh nào được phát hiện trong dữ liệu.")
            return

        with st.expander("📄 Xem bảng kết quả"):
            st.dataframe(df_out, use_container_width=True)

        with st.expander("📊 Thống kê cảm xúc"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Tỷ lệ cảm xúc")
                fig1, ax1 = plt.subplots()
                df_out['Cảm xúc'].value_counts().plot.pie(
                    autopct='%1.1f%%', ax=ax1, colors=['green', 'red', 'gray']
                )
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
