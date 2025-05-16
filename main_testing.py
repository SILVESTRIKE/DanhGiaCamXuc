import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyvi import ViTokenizer
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Định nghĩa bộ aspects cho 2 mô hình
ASPECTS_RESTAURANT = [
    'FOOD#QUALITY', 'FOOD#PRICES', 'SERVICE#GENERAL', 'AMBIENCE#GENERAL',
    'LOCATION#GENERAL', 'RESTAURANT#GENERAL', 'FOOD#STYLE&OPTIONS',
    'RESTAURANT#PRICES', 'RESTAURANT#MISCELLANEOUS', 'DRINKS#QUALITY',
    'DRINKS#PRICES', 'DRINKS#STYLE&OPTIONS'
]

ASPECTS_HOTEL = [
    'HOTEL#GENERAL', 'HOTEL#COMFORT', 'HOTEL#CLEANLINESS', 'HOTEL#DESIGN&FEATURES',
    'HOTEL#QUALITY', 'HOTEL#PRICES', 'ROOMS#GENERAL', 'ROOMS#CLEANLINESS',
    'ROOMS#DESIGN&FEATURES', 'ROOMS#COMFORT', 'ROOMS#QUALITY', 'ROOMS#PRICES',
    'ROOM_AMENITIES#GENERAL', 'ROOM_AMENITIES#CLEANLINESS', 'ROOM_AMENITIES#QUALITY',
    'ROOM_AMENITIES#DESIGN&FEATURES', 'ROOM_AMENITIES#COMFORT', 'FACILITIES#GENERAL',
    'FACILITIES#QUALITY', 'FACILITIES#DESIGN&FEATURES', 'FACILITIES#CLEANLINESS',
    'FACILITIES#COMFORT', 'FACILITIES#PRICES', 'LOCATION#GENERAL', 'SERVICE#GENERAL',
    'FOOD&DRINKS#QUALITY', 'FOOD&DRINKS#STYLE&OPTIONS', 'FOOD&DRINKS#PRICES',
    'FOOD&DRINKS#MISCELLANEOUS'
]

LABEL_ENCODER = [ 'negative', 'neutral', 'positive']

# Load model/tokenizer dựa trên model chọn
@st.cache_resource
def load_tokenizer_and_model(model_choice):
    if model_choice == "Nhà hàng":
        model_path = "./models/phobert_restaurant_model"
        aspects = ASPECTS_RESTAURANT
    else:
        model_path = "./models/phobert_hotel_model"
        aspects = ASPECTS_HOTEL

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model, aspects

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return ViTokenizer.tokenize(text.lower())


def predict(text, tokenizer, model, aspects, label_encoder):
    clean_text = preprocess_text(text)

    # Aspect keywords theo model
    if aspects == ASPECTS_RESTAURANT:
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
    else:
        aspect_keywords = {
            'HOTEL#GENERAL': ['khách sạn', 'tổng thể', 'trải nghiệm'],
            'HOTEL#COMFORT': ['thoải mái', 'yên tĩnh', 'ồn ào'],
            'HOTEL#CLEANLINESS': ['sạch sẽ', 'bẩn', 'vệ sinh'],
            'HOTEL#DESIGN&FEATURES': ['đẹp', 'thiết kế', 'kiến trúc', 'mới'],
            'HOTEL#QUALITY': ['chất lượng', 'tiêu chuẩn'],
            'HOTEL#PRICES': ['giá', 'rẻ', 'đắt', 'hợp lý'],
            'ROOMS#GENERAL': ['phòng', 'phòng ốc'],
            'ROOMS#CLEANLINESS': ['sạch', 'bẩn', 'vệ sinh phòng'],
            'ROOMS#DESIGN&FEATURES': ['rộng', 'chật', 'thiết kế phòng'],
            'ROOMS#COMFORT': ['thoải mái', 'khó chịu', 'giường'],
            'ROOMS#QUALITY': ['chất lượng phòng'],
            'ROOMS#PRICES': ['giá phòng', 'chi phí'],
            'ROOM_AMENITIES#GENERAL': ['tiện nghi', 'đồ dùng'],
            'ROOM_AMENITIES#CLEANLINESS': ['khăn sạch', 'khăn bẩn'],
            'ROOM_AMENITIES#QUALITY': ['chất lượng đồ dùng'],
            'ROOM_AMENITIES#DESIGN&FEATURES': ['thiết kế đồ dùng'],
            'ROOM_AMENITIES#COMFORT': ['thoải mái đồ dùng'],
            'FACILITIES#GENERAL': ['cơ sở', 'dịch vụ'],
            'FACILITIES#QUALITY': ['chất lượng cơ sở'],
            'FACILITIES#DESIGN&FEATURES': ['thiết kế cơ sở', 'bể bơi'],
            'FACILITIES#CLEANLINESS': ['sạch cơ sở'],
            'FACILITIES#COMFORT': ['thoải mái cơ sở'],
            'FACILITIES#PRICES': ['giá dịch vụ'],
            'LOCATION#GENERAL': ['vị trí', 'địa điểm', 'gần', 'xa'],
            'SERVICE#GENERAL': ['phục vụ', 'nhân viên', 'thái độ'],
            'FOOD&DRINKS#QUALITY': ['ngon', 'dở', 'chất lượng đồ ăn'],
            'FOOD&DRINKS#STYLE&OPTIONS': ['đa dạng', 'menu', 'lựa chọn'],
            'FOOD&DRINKS#PRICES': ['giá đồ ăn', 'rẻ', 'đắt'],
            'FOOD&DRINKS#MISCELLANEOUS': ['bữa sáng', 'buffet']
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
                "label": LABEL_ENCODER[label_id],
                "probs": label_probs
            }
    return results, len(text), input_ids.shape[1], len(clean_text.split()), datetime.datetime.now()


# Khởi tạo session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'selected_history_index' not in st.session_state:
    st.session_state.selected_history_index = None

st.set_page_config(page_title="Phân tích cảm xúc đa mô hình", layout="wide")
tabs = st.tabs(["🔍 Nhập liệu & Kết quả", "📈 Biểu đồ", "📜 Lịch sử", "📊 Thống kê"])

with tabs[0]:
    st.title("🔍 Nhập liệu & Kết quả")

    # Model selection
    model_choice = st.selectbox("Chọn mô hình phân tích", ["Nhà hàng", "Khách sạn"])

    tokenizer, model, aspects = load_tokenizer_and_model(model_choice)

    if st.session_state.selected_history_index is not None:
        entry = st.session_state.history[st.session_state.selected_history_index]
        text_input = entry['text']
        results = entry['result']
        char_len = entry['char_len']
        token_len = entry['token_len']
        word_count = entry['word_count']
        timestamp = entry['time']
        saved_model_choice = entry['model_choice']
        st.session_state.selected_history_index = None
    else:
        text_input = st.text_area("Nhập đánh giá của bạn:", height=150)
        results = None
        saved_model_choice = model_choice

    if st.button("Phân tích", key="analyze") and results is None:
        if not text_input.strip():
            st.warning("Vui lòng nhập nội dung để phân tích.")
        else:
            prediction = predict(text_input, tokenizer, model, aspects, LABEL_ENCODER)
            if isinstance(prediction, str):
                st.warning(prediction)
            else:
                results, char_len, token_len, word_count, timestamp = prediction
                saved_model_choice = model_choice
                st.session_state.history.append({
                    "text": text_input,
                    "result": results,
                    "char_len": char_len,
                    "token_len": token_len,
                    "word_count": word_count,
                    "time": timestamp,
                    "model_choice": saved_model_choice
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

        colors = {"POSITIVE": ("#ccffcc", "#006600"), "NEGATIVE": ("#ffcccc", "#660000"), "NEUTRAL": ("#e6e6e6", "#333333")}.get(final_sentiment.upper(), ("#ffffff", "#000000"))
        bg_color, text_color = colors
        st.markdown(f'<div style="background-color: {bg_color}; color: {text_color}; border: 1px solid #ccc; padding: 10px; border-radius: 5px;">Tổng kết: Đánh giá toàn câu là <b>{final_sentiment.upper()}</b> với độ tự tin khoảng <b>{confidence:.2f}</b>, điểm đánh giá ước lượng: <b>{score}/10</b></div>', unsafe_allow_html=True)
        st.markdown(f"**Độ dài văn bản:** {char_len} ký tự | {word_count} từ | {token_len} token sau mã hóa")

with tabs[1]:
    st.title("📈 Biểu đồ tổng quan")

    if not st.session_state.history:
        st.info("Chưa có dữ liệu để hiển thị biểu đồ.")
    else:
        latest = st.session_state.history[-1]
        tokenizer, model, aspects = load_tokenizer_and_model(latest['model_choice'])

        df = pd.DataFrame([
            [aspect, detail['label'], *detail['probs']] for aspect, detail in latest['result'].items()
        ], columns=["Khía cạnh", "Cảm xúc", "Tích cực", "Tiêu cực", "Trung tính"])

        st.markdown(f"**Độ dài văn bản:** {latest['char_len']} ký tự | {latest['word_count']} từ | {latest['token_len']} token")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Tỷ lệ cảm xúc")
            sentiments = df['Cảm xúc'].value_counts()
            fig1, ax1 = plt.subplots()
            ax1.pie(sentiments, labels=sentiments.index, autopct ='%1.1f%%', colors=sns.color_palette('pastel'))
            st.pyplot(fig1)

        with col2:
            st.markdown("### 🧩 Biểu đồ khía cạnh")
            fig2, ax2 = plt.subplots()
            sns.countplot(x="Cảm xúc", data=df, palette='viridis', ax=ax2)
            st.pyplot(fig2)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### 🧪 Phân phối token")
            token_lengths = [entry['token_len'] for entry in st.session_state.history if entry['model_choice'] == latest['model_choice']]
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
            if st.button(f"Xem lại [{entry['time'].strftime('%Y-%m-%d %H:%M:%S')}] [{entry['model_choice']}] {entry['text'][:40]}...", key=f"load_{i}"):
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

        latest_model_choice = st.session_state.history[-1]['model_choice']

        times = [item['time'] for item in st.session_state.history if item['model_choice'] == latest_model_choice]
        sentiments_all = []
        for item in st.session_state.history:
            if item['model_choice'] == latest_model_choice:
                for aspect, detail in item['result'].items():
                    sentiments_all.append({"time": item['time'], "label": detail['label']})

        df_all = pd.DataFrame(sentiments_all)
        if not df_all.empty:
            df_all['date'] = df_all['time'].dt.date
            st.markdown(f"### 📅 Biểu đồ cảm xúc theo thời gian cho model **{latest_model_choice}**")
            fig_time, ax_time = plt.subplots()
            sns.countplot(x="date", hue="label", data=df_all, ax=ax_time)
            ax_time.set_ylabel("Số lượt")
            st.pyplot(fig_time)

        st.markdown(f"### ⚙️ Thông số mô hình cho model **{latest_model_choice}**")
        st.markdown(f"- Mô hình: `{latest_model_choice}`")
        st.markdown(f"- Tổng số khía cạnh: {len(ASPECTS_RESTAURANT) if latest_model_choice=='Nhà hàng' else len(ASPECTS_HOTEL)}")
        st.markdown(f"- Nhãn: {', '.join(LABEL_ENCODER)}")
        st.markdown(f"- Thời gian phân tích đầu tiên: {min(times).strftime('%Y-%m-%d %H:%M:%S') if times else 'Chưa có dữ liệu'}")
        st.markdown(f"- Thời gian gần nhất: {max(times).strftime('%Y-%m-%d %H:%M:%S') if times else 'Chưa có dữ liệu'}")
