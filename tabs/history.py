import streamlit as st

def history():
    st.header("📜 Lịch sử phân tích")

    if 'history' not in st.session_state:
        st.session_state.history = []

    if st.button("🗑 Xoá toàn bộ lịch sử"):
        st.session_state.history.clear()
        st.rerun()

    for idx, entry in enumerate(reversed(st.session_state.history)):
        i = len(st.session_state.history) - 1 - idx
        col1, col2 = st.columns([8, 2])
        with col1:
            if st.button(f"Xem lại [{entry['model_choice']}] {entry['text'][:40]}...", key=f"load_{i}"):
                st.session_state.selected_history_index = i
                st.rerun()
        with col2:
            if st.button("Xoá", key=f"del_{i}"):
                st.session_state.history.pop(i)
                st.rerun()