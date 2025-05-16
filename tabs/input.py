import streamlit as st
import pandas as pd
from utils import load_model_and_tokenizer, predict
from config import ASPECTS

def input():
    model_choice = st.session_state.model_choice
    tokenizer, model = load_model_and_tokenizer(model_choice)

    if 'history' not in st.session_state:
        st.session_state.history = []

    text_input = st.text_area("Nhập đánh giá:", height=150)
    if st.button("Phân tích") and text_input.strip():
        output = predict(text_input, tokenizer, model, aspect_type=model_choice)

        if isinstance(output, str):
            st.warning(output)
        else:
            results, char_len, token_len, word_count, timestamp = output
            st.session_state.history.append({
                "text": text_input,
                "result": results,
                "char_len": char_len,
                "token_len": token_len,
                "word_count": word_count,
                "time": timestamp,
                "model_choice": model_choice
            })

    if st.session_state.history:
        latest = st.session_state.history[-1]
        df = pd.DataFrame([
            [k, v['label'], *v['probs']] for k, v in latest['result'].items()
        ], columns=["Khía cạnh", "Cảm xúc", "Tích cực", "Tiêu cực", "Trung tính"])
        st.dataframe(df)

        # Lấy nhãn phổ biến nhất trong kết quả
        most_common_sentiment = df['Cảm xúc'].value_counts().idxmax()
        confidence = df[['Tích cực','Tiêu cực','Trung tính']].max(axis=1).mean()

        # Map nhãn sang màu sắc
        color_map = {
            'positive': 'green',
            'negative': 'red',
            'neutral': 'gray'
        }
        color = color_map.get(most_common_sentiment.lower(), 'black')

        st.markdown(
            f'<p style="color:{color}; font-weight:bold;">'
            f'Tổng quan: {most_common_sentiment} | Tự tin: {confidence:.2f}'
            f'</p>',
            unsafe_allow_html=True
        )
        st.markdown(f"**Văn bản:** {latest['char_len']} ký tự | {latest['word_count']} từ | {latest['token_len']} token")
