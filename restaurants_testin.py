import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Định nghĩa global biến để dùng chung
aspects = [
    'FOOD#QUALITY', 'FOOD#PRICES', 'SERVICE#GENERAL', 'AMBIENCE#GENERAL',
    'LOCATION#GENERAL', 'RESTAURANT#GENERAL', 'FOOD#STYLE&OPTIONS',
    'RESTAURANT#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY',
    'DRINKS#PRICES', 'DRINKS#STYLE&OPTIONS'
]

label_encoder = ['positive', 'negative', 'neutral']

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(".\models\phobert_restaurant_model")

# Load model
@st.cache_resource
def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained(".\models\phobert_restaurant_model")
        model.eval()
        return model
    except Exception:
        st.error("Không thể load model. Đảm bảo file trọng số đã được đặt đúng thư mục.")
        return None

# Preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = word_tokenize(text, format="text").lower()
    return text

# Predict function
def predict(text, tokenizer, model):
    clean_text = preprocess_text(text)
    aspect_keywords = {
        'FOOD#QUALITY': ['ngon', 'dở', 'tươi', 'chất lượng', 'vị', 'thức ăn', 'đồ ăn'],
        'FOOD#PRICES': ['giá', 'rẻ', 'đắt', 'tiền', 'chi phí'],
        'SERVICE#GENERAL': ['phục vụ', 'nhân viên', 'nhanh', 'chậm', 'thái độ'],
        'AMBIENCE#GENERAL': ['không gian', 'view', 'sạch', 'bẩn', 'thoáng', 'đẹp'],
        'LOCATION#GENERAL': ['vị trí', 'địa điểm', 'dễ tìm', 'xa', 'gần'],
        'RESTAURANT#GENERAL': ['quán', 'nhà hàng', 'tổng thể', 'trải nghiệm'],
        'FOOD#STYLE&OPTIONS': ['món', 'menu', 'đa dạng', 'lựa chọn', 'kiểu'],
        'RESTAURANT#PRICES': ['giá cả', 'chi phí', 'hợp lý', 'đắt đỏ'],
        'RESTAURANT#MISCELLANEOUS': ['gửi xe', 'vệ sinh', 'khăn lạnh', 'tiện ích'],
        'DRINKS#QUALITY': ['nước uống', 'trà', 'nước ngọt', 'đồ uống'],
        'DRINKS#PRICES': ['giá', 'rẻ', 'đắt', 'nước uống'],
        'DRINKS#STYLE&OPTIONS': ['nước uống', 'menu', 'đa dạng']
    }

    mentioned_aspects = [aspect for aspect, keywords in aspect_keywords.items() if any(keyword in clean_text for keyword in keywords)]
    if not mentioned_aspects:
        return "Không phát hiện khía cạnh nào liên quan trong câu."

    encoding = tokenizer.encode_plus(
        clean_text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits.view(-1, len(aspects), 3)
    probs = torch.nn.functional.softmax(logits, dim=2)[0]
    preds = torch.argmax(probs, dim=1)

    results = {}
    for i, aspect in enumerate(aspects):
        if aspect in mentioned_aspects:
            label_id = preds[i].item()
            label_probs = probs[i].tolist()
            results[aspect] = {
                "label": label_encoder[label_id],
                "probs": label_probs
            }
    return results, len(text), input_ids.shape[1], len(clean_text.split()), datetime.datetime.now()

# Khởi tạo session_state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'selected_history_index' not in st.session_state:
    st.session_state.selected_history_index = None

st.set_page_config(page_title="Phân tích cảm xúc nhà hàng", layout="wide")
tabs = st.tabs(["🔍 Nhập liệu & Kết quả", "📈 Biểu đồ", "📜 Lịch sử", "📊 Thống kê"])

with tabs[0]:
    st.title("🔍 Nhập liệu & Kết quả")

    if st.session_state.selected_history_index is not None:
        entry = st.session_state.history[st.session_state.selected_history_index]
        text_input = entry['text']
        results = entry['result']
        char_len = entry['char_len']
        token_len = entry['token_len']
        word_count = entry['word_count']
        timestamp = entry['time']
        st.session_state.selected_history_index = None
    else:
        text_input = st.text_area("Nhập đánh giá của bạn:", height=150)
        results = None

    if st.button("Phân tích", key="analyze") and results is None:
        if not text_input.strip():
            st.warning("Vui lòng nhập nội dung để phân tích.")
        else:
            tokenizer = load_tokenizer()
            model = load_model()

            prediction = predict(text_input, tokenizer, model)
            if isinstance(prediction, str):
                st.warning(prediction)
            else:
                results, char_len, token_len, word_count, timestamp = prediction

                st.session_state.history.append({
                    "text": text_input,
                    "result": results,
                    "char_len": char_len,
                    "token_len": token_len,
                    "word_count": word_count,
                    "time": timestamp
                })

    if results:
        st.subheader("Kết quả phân tích")
        rows = []
        for aspect, detail in results.items():
            rows.append([aspect, detail['label'], *detail['probs']])
        df = pd.DataFrame(rows, columns=["Khía cạnh", "Cảm xúc", "Tích cực", "Tiêu cực", "Trung tính"])
        st.dataframe(df)

        final_counts = df['Cảm xúc'].value_counts()
        final_sentiment = final_counts.idxmax() if not final_counts.empty else "neutral"
        confidence = df[["Tích cực", "Tiêu cực", "Trung tính"]].max(axis=1).mean()
        score = round(confidence * 10, 2)

        color = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"}.get(final_sentiment.upper(), "black")
        st.markdown(f'<div style="color: {color};">Tổng kết: Đánh giá toàn câu là <b>{final_sentiment.upper()}</b> với độ tự tin khoảng <b>{confidence:.2f}</b>, điểm đánh giá ước lượng: <b>{score}/10</b></div>', unsafe_allow_html=True)
        st.markdown(f"**Độ dài văn bản:** {char_len} ký tự | {word_count} từ | {token_len} token sau mã hóa")

with tabs[1]:
    st.title("📈 Biểu đồ tổng quan")
    if not st.session_state.history:
        st.info("Chưa có dữ liệu để hiển thị biểu đồ.")
    else:
        latest = st.session_state.history[-1]
        df = pd.DataFrame([
            [aspect, detail['label'], *detail['probs']] for aspect, detail in latest['result'].items()
        ], columns=["Khía cạnh", "Cảm xúc", "Tích cực", "Tiêu cực", "Trung tính"])

        st.markdown(f"**Độ dài văn bản:** {latest['char_len']} ký tự | {latest['word_count']} từ | {latest['token_len']} token")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Tỷ lệ cảm xúc")
            sentiments = df['Cảm xúc'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(sentiments, labels=sentiments.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
            st.pyplot(fig1)

        with col2:
            st.markdown("### 🧩 Biểu đồ khía cạnh")
            fig2, ax2 = plt.subplots()
            sns.countplot(x="Cảm xúc", data=df, palette='viridis', ax=ax2)
            st.pyplot(fig2)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### 🧪 Phân phối token")
            token_lengths = [entry['token_len'] for entry in st.session_state.history]
            fig3, ax3 = plt.subplots()
            sns.histplot(token_lengths, bins=10, kde=True, ax=ax3)
            st.pyplot(fig3)

        with col4:
            st.markdown("### 📉 Biểu đồ xác suất trung bình")
            mean_probs = df[["Tích cực", "Tiêu cực", "Trung tính"]].mean()
            fig4, ax4 = plt.subplots()
            sns.barplot(x=mean_probs.index, y=mean_probs.values, palette='pastel', ax=ax4)
            st.pyplot(fig4)

with tabs[2]:
    st.title("📜 Lịch sử tương tác")
    if st.button("🗑️ Xóa toàn bộ lịch sử"):
        st.session_state.history = []
        st.success("Đã xóa lịch sử.")
        st.experimental_rerun()

    for idx, entry in enumerate(reversed(st.session_state.history)):
        i = len(st.session_state.history) - 1 - idx
        col1, col2 = st.columns([8, 2])
        with col1:
            if st.button(f"Xem lại [{entry['time'].strftime('%Y-%m-%d %H:%M:%S')}] {entry['text'][:40]}...", key=f"load_{i}"):
                st.session_state.selected_history_index = i
                st.experimental_rerun()
        with col2:
            if st.button("Xoá", key=f"delete_{i}"):
                st.session_state.history.pop(i)
                st.experimental_rerun()

with tabs[3]:
    st.title("📊 Thống kê tổng hợp")
    if not st.session_state.history:
        st.info("Chưa có dữ liệu để thống kê.")
    else:
        total = len(st.session_state.history)
        st.metric("Tổng số lượt đánh giá", total)

        times = [item['time'] for item in st.session_state.history]
        sentiments_all = []
        for item in st.session_state.history:
            for aspect, detail in item['result'].items():
                sentiments_all.append({"time": item['time'], "label": detail['label']})

        df_all = pd.DataFrame(sentiments_all)
        df_all['date'] = df_all['time'].dt.date

        st.markdown("### 📅 Biểu đồ cảm xúc theo thời gian")
        fig_time, ax_time = plt.subplots()
        sns.countplot(x="date", hue="label", data=df_all, ax=ax_time)
        ax_time.set_ylabel("Số lượt")
        st.pyplot(fig_time)

        st.markdown("### ⚙️ Thông số mô hình")
        st.markdown("- Mô hình: `phobert-base`")
        st.markdown(f"- Tổng số khía cạnh: {len(aspects)}")
        st.markdown(f"- Nhãn: {', '.join(label_encoder)}")
        st.markdown(f"- Thời gian phân tích đầu tiên: {min(times).strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(f"- Thời gian gần nhất: {max(times).strftime('%Y-%m-%d %H:%M:%S')}")
