import numpy as np
from transformers import TFBertForSequenceClassification
import joblib

def load_model_and_tokenizer(save_directory='./bert_model'):
    model = TFBertForSequenceClassification.from_pretrained(save_directory)
    tokenizer = BertTokenizerFast.from_pretrained(save_directory)
    label_encoder = joblib.load(os.path.join(save_directory, 'label_encoder.pkl'))
    return model, tokenizer, label_encoder

def predict_category(model, tokenizer, label_encoder, subcategory, note, amount):
    text = f"Subcategory: {subcategory} Note: {note} Amount: {amount}"
    encoded_input = tokenizer(
        text=text,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors="tf"
    )
    prediction = model(encoded_input)
    predicted_category = label_encoder.inverse_transform(np.argmax(prediction.logits, axis=1))
    return predicted_category[0]
