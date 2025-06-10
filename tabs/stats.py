import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import ASPECTS, LABEL_ENCODER

def stats():
    model_choice = st.session_state.model_choice
    data = [x for x in st.session_state.get("history", []) if x['model_choice'] == model_choice]

    if not data:
        st.info("Chưa có dữ liệu để thống kê.")
        return

    st.metric("Tổng số lượt đánh giá", len(data))

    times = [x['time'] for x in data]
    sentiments_all = []
    for item in data:
        result = item.get('result', {})
        if isinstance(result, dict):
            for asp, detail in result.items():
                label = detail.get('label')
                if label:
                    sentiments_all.append({"time": item['time'], "label": label.capitalize()})

    if not sentiments_all:
        st.warning("Không có dữ liệu cảm xúc để thống kê.")
        return

    df_all = pd.DataFrame(sentiments_all)
    df_all['date'] = pd.to_datetime(df_all['time']).dt.date

    st.markdown("### Biểu đồ cảm xúc theo thời gian")
    fig_time, ax_time = plt.subplots(figsize=(8, 4))
    sns.countplot(x="date", hue="label", data=df_all, palette={
        'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'
    }, ax=ax_time)
    ax_time.set_ylabel("Số lượt")
    ax_time.set_xlabel("Ngày")
    ax_time.set_title("Phân bố cảm xúc theo ngày")
    st.pyplot(fig_time)

    st.markdown("### ⚙️ Thông số mô hình")
    st.markdown(f"- **Mô hình đang sử dụng:** `{model_choice}`")
    st.markdown(f"- **Tổng số khía cạnh:** {len(ASPECTS[model_choice])}")
    st.markdown(f"- **Nhãn cảm xúc:** {', '.join([x.capitalize() for x in LABEL_ENCODER])}")
    st.markdown(f"- **Thời gian đầu tiên:** {min(times).strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"- **Lần gần nhất:** {max(times).strftime('%Y-%m-%d %H:%M:%S')}")
