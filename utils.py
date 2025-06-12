import re
import datetime
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from underthesea import word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import ASPECTS, LABEL_ENCODER, ASPECT_KEYWORDS
import streamlit as st
import unicodedata

@st.cache_resource
def load_model_and_tokenizer(model_choice):
    path_map = {
        "Nhà hàng": "HakuDevon/phobert_restaurant_model",
    }
    path = path_map[model_choice]
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.eval()
        aspects = ASPECTS[model_choice]
        expected_size = len(aspects) * 3
        test_input = tokenizer.encode_plus(
            "test", max_length=128, padding='max_length', truncation=True, return_tensors='pt'
        )
        with torch.no_grad():
            test_output = model(test_input['input_ids'], attention_mask=test_input['attention_mask'])
        if test_output.logits.shape[-1] != expected_size:
            st.error(f"Model {model_choice} không hỗ trợ {len(aspects)} khía cạnh. Kích thước đầu ra: {test_output.logits.shape[-1]}, kỳ vọng: {expected_size}")
            return None, None
        return tokenizer, model
    except Exception as e:
        st.error(f"Lỗi khi tải model {model_choice}: {e}")
        return None, None


class DummyPreprocessor:
    def process_text(self, text, normalize_tone=True, segment=True):
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[^\w\s.,?!]', '', text)
        return word_tokenize(text, format="text").lower()


def predict(text, tokenizer, model, aspect_type, device=None, preprocessor=None, confidence_threshold=None):
    aspects = ASPECTS[aspect_type]
    keywords_dict = ASPECT_KEYWORDS[aspect_type]

    if preprocessor is None:
        preprocessor = DummyPreprocessor()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if confidence_threshold is None:
        confidence_threshold = st.session_state.get("confidence_threshold", 0.7)


    clean_text = preprocessor.process_text(text, normalize_tone=True, segment=True)
    mentioned_aspects = [
        aspect for aspect, keywords in keywords_dict.items()
        if any(keyword in clean_text for keyword in keywords)
    ]

    if not clean_text.strip():
        return "Vui lòng nhập một câu có nội dung."

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

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits.view(-1, len(aspects), 3)
    label_encoder = LabelEncoder()
    label_encoder.fit(LABEL_ENCODER)

    results = {}
    for i, aspect in enumerate(aspects):
        probs = F.softmax(logits[0, i, :], dim=0)
        confidence, predicted_idx = torch.max(probs, dim=0)
        label = label_encoder.inverse_transform([predicted_idx.item()])[0]

        if aspect in mentioned_aspects or (label != 'neutral' and confidence.item() > confidence_threshold):
            results[aspect] = {
                "label": label,
                "probs": probs.tolist()
            }

    return results if results else "Không xác định được khía cạnh nào trong câu.", len(text), input_ids.shape[1], len(clean_text.split()), datetime.datetime.now()
