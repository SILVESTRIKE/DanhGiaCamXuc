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
        for asp, detail in item['result'].items():
            sentiments_all.append({"time": item['time'], "label": detail['label']})

    df_all = pd.DataFrame(sentiments_all)
    df_all['date'] = df_all['time'].dt.date

    st.markdown(f"### 📅 Biểu đồ cảm xúc theo thời gian")
    fig_time, ax_time = plt.subplots()
    sns.countplot(x="date", hue="label", data=df_all, ax=ax_time)
    ax_time.set_ylabel("Số lượt")
    st.pyplot(fig_time)

    st.markdown(f"### ⚙️ Thông số mô hình")
    st.markdown(f"- Mô hình: `{model_choice}`")
    st.markdown(f"- Tổng số khía cạnh: {len(ASPECTS[model_choice])}")
    st.markdown(f"- Nhãn: {', '.join(LABEL_ENCODER)}")
    st.markdown(f"- Thời gian đầu tiên: {min(times).strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"- Gần nhất: {max(times).strftime('%Y-%m-%d %H:%M:%S')}")
