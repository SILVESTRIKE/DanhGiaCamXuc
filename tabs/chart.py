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
        [k, v['label'].capitalize(), *v['probs']] for k, v in latest['result'].items()
    ], columns=["Khía cạnh", "Cảm xúc", "Negative", "Neutral", "Positive"])

    # Tỷ lệ cảm xúc (pie chart)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Tỷ lệ cảm xúc")
        fig1, ax1 = plt.subplots()
        df['Cảm xúc'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax1, colors=['red', 'gray', 'green'])
        ax1.set_ylabel("")
        ax1.set_title("Tỷ lệ các cảm xúc")
        st.pyplot(fig1)

    # Phân bố cảm xúc (bar chart)
    with col2:
        st.markdown("### Phân bố cảm xúc")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x="Cảm xúc", palette={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}, ax=ax2)
        ax2.set_title("Số lượng cảm xúc theo khía cạnh")
        st.pyplot(fig2)

    # Phân phối token
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Phân phối token")
        token_lengths = [entry['token_len'] for entry in data]
        fig3, ax3 = plt.subplots()
        sns.histplot(token_lengths, bins=10, kde=True, ax=ax3, color='skyblue')
        ax3.set_xlabel("Số token")
        ax3.set_title("Phân phối độ dài token")
        st.pyplot(fig3)

    # Xác suất trung bình mỗi nhãn
    with col4:
        st.markdown("### Xác suất trung bình cảm xúc")
        mean_probs = df[["Negative", "Neutral", "Positive"]].mean()
        fig4, ax4 = plt.subplots()
        sns.barplot(x=mean_probs.index, y=mean_probs.values, palette=['red', 'gray', 'green'], ax=ax4)
        ax4.set_ylabel("Xác suất trung bình")
        ax4.set_title("Xác suất trung bình mỗi cảm xúc")
        st.pyplot(fig4)
