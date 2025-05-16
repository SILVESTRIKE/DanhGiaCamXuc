import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def chart():
    model_choice = st.session_state.model_choice
    data = [x for x in st.session_state.get("history", []) if x['model_choice'] == model_choice]

    if not data:
        st.info("Chưa có dữ liệu.")
        return

    latest = data[-1]
    df = pd.DataFrame([
        [k, v['label'], *v['probs']] for k, v in latest['result'].items()
    ], columns=["Khía cạnh", "Cảm xúc", "Tích cực", "Tiêu cực", "Trung tính"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Tỷ lệ cảm xúc")
        fig1, ax1 = plt.subplots()
        df['Cảm xúc'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)

    with col2:
        st.markdown("### 📊 Phân bố cảm xúc")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x="Cảm xúc", palette='Set2', ax=ax2)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 🔠 Phân phối token")
        token_lengths = [entry['token_len'] for entry in data]
        fig3, ax3 = plt.subplots()
        sns.histplot(token_lengths, bins=10, kde=True, ax=ax3)
        st.pyplot(fig3)

    with col4:
        st.markdown("### 📉 Xác suất trung bình")
        mean_probs = df[["Tích cực", "Tiêu cực", "Trung tính"]].mean()
        fig4, ax4 = plt.subplots()
        sns.barplot(x=mean_probs.index, y=mean_probs.values, palette='pastel', ax=ax4)
        st.pyplot(fig4)