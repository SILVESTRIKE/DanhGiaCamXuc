import streamlit as st
import pandas as pd
from utils import load_model_and_tokenizer, predict, DummyPreprocessor
from config import ASPECTS, LABEL_ENCODER
import torch

def input():
    model_choice = st.session_state.model_choice
    tokenizer, model = load_model_and_tokenizer(model_choice)

    if 'history' not in st.session_state:
        st.session_state.history = []

    text_input = st.text_area("Nhập đánh giá:", height=200)
    if st.button("Phân tích") and text_input.strip():
        output = predict(
            text_input,
            tokenizer,
            model,
            aspect_type=model_choice,
            device=None,
            preprocessor=DummyPreprocessor(),
            confidence_threshold=st.session_state.confidence_threshold
        )
    

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
        result_data = latest['result']

        if isinstance(result_data, dict):
            # Hiển thị kết quả theo format cũ: label + probs
            df = pd.DataFrame([
                [k, v['label'].capitalize(), f"{max(v['probs']) * 100:.1f}%"]
                for k, v in result_data.items()
            ], columns=["Khía cạnh", "Cảm xúc", "Độ tin cậy"])

            st.dataframe(df, use_container_width=True)

            # Thống kê tổng quan
            most_common_sentiment = df['Cảm xúc'].value_counts().idxmax()
            avg_confidence = df['Độ tin cậy'].apply(lambda x: float(x.strip('%'))).mean() / 100

            color_map = {
                'Positive': 'green',
                'Negative': 'red',
                'Neutral': 'gray'
            }
            color = color_map.get(most_common_sentiment, 'black')

            st.markdown(
                f'<p style="color:{color}; font-weight:bold; font-size:18px;">'
                f'Tổng quan: {most_common_sentiment} | Tự tin trung bình: {avg_confidence:.2%}'
                f'</p>',
                unsafe_allow_html=True
            )
        else:
            st.warning(result_data)

        st.markdown(
            f"**Văn bản:** {latest['char_len']} ký tự | {latest['word_count']} từ | {latest['token_len']} token"
        )
