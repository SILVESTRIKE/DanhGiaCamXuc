# Ứng Dụng PhoBERT Trong Phân Tích Cảm Xúc Đa Khía Cạnh: Đánh Giá Nhà Hàng Tiếng Việt

Đồ án học phần Deep Learning – Nhóm 8 – Đại học Công Thương TP.HCM

## Mô tả dự án

Dự án tập trung vào phân tích cảm xúc đa khía cạnh (ABSA) trong các đánh giá nhà hàng tiếng Việt, sử dụng mô hình PhoBERT được fine-tune để dự đoán cảm xúc (tích cực, tiêu cực, trung lập) theo từng khía cạnh cụ thể như chất lượng đồ ăn, dịch vụ, không gian,... Dự đoán được hiển thị trên ứng dụng web bằng Streamlit.

## Thành viên thực hiện

| MSSV         | Họ Tên             | Vai trò chính                                        |
|--------------|--------------------|------------------------------------------------------|
| 2001220817   | Văn Trọng Dương    | Phát triển ứng dụng Streamlit                        |
| 2001223664   | Lương Liêm Phong   | Trưởng nhóm, huấn luyện mô hình PhoBERT             |
| 2001225914   | Trần Khánh Vũ      | Thu thập và xử lý dữ liệu                            |

## Mục tiêu chính

- Huấn luyện mô hình PhoBERT để phân tích cảm xúc theo 12 khía cạnh nhà hàng.
- Triển khai giao diện người dùng với Streamlit để nhập đánh giá và hiển thị kết quả dự đoán.
- Ứng dụng mô hình vào tình huống thực tế nhằm hỗ trợ doanh nghiệp cải thiện dịch vụ.

## Công nghệ sử dụng

- Mô hình: PhoBERT (vinai/phobert-base), Transformers, PyTorch
- Xử lý tiếng Việt: underthesea
- Giao diện: Streamlit, Pandas, Seaborn, Matplotlib
- Môi trường huấn luyện: Google Colab (GPU)
- Lưu trữ mô hình: Google Drive

## Chức năng chính

- Phân tích cảm xúc theo từng khía cạnh trong đánh giá
- Giao diện web gồm 4 tab: Nhập liệu, Thống kê, Lịch sử, Biểu đồ
- Quản lý lịch sử phân tích, hiển thị biểu đồ trực quan

## Cách cài đặt

1. Cài đặt Python >= 3.8
2. Cài các thư viện cần thiết:

   pip install transformers torch underthesea sklearn tqdm streamlit pandas matplotlib seaborn

3. Chạy ứng dụng Streamlit:

   streamlit run app.py

## Hướng dẫn sử dụng

* Chọn mô hình và nhập đoạn đánh giá vào tab "Input"
* Nhấn nút "Phân tích" để xem kết quả theo khía cạnh
* Chuyển sang các tab "Stats", "Chart", "History" để xem thống kê, biểu đồ và lịch sử phân tích

## Kết quả huấn luyện

* Accuracy kiểm thử đạt: 0.8025
* Một số khía cạnh đạt độ chính xác cao:

  * DRINKS#PRICES: 93.8%
  * DRINKS#STYLE\&OPTIONS: 91.6%
  * FOOD#QUALITY: 79.9%
  * SERVICE#GENERAL: 76.6%

## Định hướng phát triển

* Tăng cường dữ liệu và cân bằng khía cạnh
* Thử nghiệm các mô hình lớn hơn như phobert-large, ViBERT
* Xuất báo cáo PDF, phân tích hàng loạt từ CSV
* Triển khai ứng dụng trên nền tảng cloud

## Tài liệu tham khảo

* Devlin et al. (2018), BERT
* Nguyen et al. (2020), PhoBERT
* VLSP 2018 dataset
* Hugging Face Transformers
* Streamlit Docs

---

Bản quyền thuộc Nhóm 8 – Khoa Công Nghệ Thông Tin – Đại học Công Thương TP.HCM – 2025

---

