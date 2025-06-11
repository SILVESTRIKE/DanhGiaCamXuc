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
                encodings = ['utf-8', 'latin1', 'utf-8-sig']
                delimiters = [',', ';', '\t']
                df = None
                for enc in encodings:
                    for delim in delimiters:
                        try:
                            df = pd.read_csv(uploaded_file, encoding=enc, sep=delim)
                            break
                        except Exception:
                            continue
                    if df is not None:
                        break
                if df is None:
                    st.error("Không thể đọc file CSV với bất kỳ mã hóa hoặc dấu phân cách nào.")
                    return
                # Check line count
                with io.TextIOWrapper(uploaded_file, encoding='utf-8') as f:
                    raw_lines = len(f.readlines())
                st.write(f"Số dòng thô: {raw_lines}, Số dòng đọc được: {len(df)}")
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
                
        # 1. Tạo một placeholder (khung chứa) rỗng
        status_placeholder = st.empty()

        # 2. Đặt thanh progress bar vào placeholder đó
        # Dùng `with` để các thành phần con được gom vào placeholder
        with status_placeholder.container():
            st.write(f"Đang phân tích {len(df)} dòng đánh giá...")
            progress_bar = st.progress(0)

        # 3. Chạy vòng lặp và chỉ cập nhật thanh progress bar
        for idx, row in enumerate(df.itertuples(index=False)):
            try:
                text = str(row.review)
                if not text.strip():
                    continue
            except AttributeError:
                status_placeholder.error("Lỗi đọc cột 'review'. Vui lòng kiểm tra lại tên cột trong file.")
                st.stop()

            output = predict(text, tokenizer, model, aspect_type=model_choice, preprocessor=preprocessor)

            if isinstance(output, str) or not isinstance(output, (list, tuple)) or len(output) == 0:
                continue
            
            result, *_ = output
            for aspect, val in result.items():
                results_all.append({
                    'STT': str(idx + 1),
                    'Văn bản': text,
                    'Khía cạnh': aspect,
                    'Cảm xúc': val['label'].capitalize(),
                    'Tự tin': max(val['probs'])
                })
            
            # Cập nhật thanh progress bên trong placeholder
            progress_bar.progress((idx + 1) / len(df))

        # 4. Sau khi xong, cập nhật chính placeholder đó bằng thông báo thành công
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