import re
import datetime
import torch
from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import ASPECTS, LABEL_ENCODER, ASPECT_KEYWORDS
import streamlit as st

@st.cache_resource
def load_model_and_tokenizer(model_choice):
    path_map = {
        "Nhà hàng": "HakuDevon/phobert_restaurant_model",
        "Khách sạn": "HakuDevon/phobert_hotel_model"
    }
    path = path_map[model_choice]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

# @st.cache_resource
# def load_model_and_tokenizer(model_choice):
#     path = "./models/phobert_restaurant_model" if model_choice == "Nhà hàng" else "./models/phobert_hotel_model"
#     tokenizer = AutoTokenizer.from_pretrained(path)
#     model = AutoModelForSequenceClassification.from_pretrained(path)
#     model.eval()
#     return tokenizer, model

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return ViTokenizer.tokenize(text.lower())

def predict(text, tokenizer, model, aspect_type):  # aspect_type = "Nhà hàng" hoặc "Khách sạn"
    aspects = ASPECTS[aspect_type]
    keywords_dict = ASPECT_KEYWORDS[aspect_type]
    
    clean_text = preprocess_text(text)
    mentioned_aspects = [
        aspect for aspect, keywords in keywords_dict.items()
        if any(keyword in clean_text for keyword in keywords)
    ]

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
            results[aspect] = {
                "label": LABEL_ENCODER[label_id],
                "probs": probs[i].tolist()
            }

    return results, len(text), input_ids.shape[1], len(clean_text.split()), datetime.datetime.now()