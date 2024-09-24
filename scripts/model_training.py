import numpy as np
import tensorflow as tf
from transformers import BertConfig, TFBertForSequenceClassification, TFBertForPreTraining
from sklearn.model_selection import train_test_split

# Function to fine-tune BERT model
def fine_tune_model(df, custom_tokenizer, label_encoder):
    def encode_features(row):
        return custom_tokenizer(
            text=f"Subcategory: {row['Subcategory']} Note: {row['Note']} Amount: {row['Amount']}",
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors="tf"
        )

    encoded_features = df.apply(encode_features, axis=1)

    input_ids = np.array([ef['input_ids'].numpy()[0] for ef in encoded_features])
    attention_mask = np.array([ef['attention_mask'].numpy()[0] for ef in encoded_features])
    y = df['category_encoded'].values

    input_ids_train, input_ids_val, attention_mask_train, attention_mask_val, y_train, y_val = train_test_split(
        input_ids, attention_mask, y, test_size=0.2, random_state=42
    )

    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=len(df['Category'].unique()))
    pretrained_model = TFBertForPreTraining.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
    model.bert.set_weights(pretrained_model.bert.get_weights())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(
        {'input_ids': input_ids_train, 'attention_mask': attention_mask_train},
        y_train,
        validation_data=({'input_ids': input_ids_val, 'attention_mask': attention_mask_val}, y_val),
        epochs=100,
        batch_size=16
    )

    return model, history

# Save model and tokenizer
def save_model_and_tokenizer(model, tokenizer, label_encoder, scaler, save_directory='./bert_model'):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    joblib.dump(label_encoder, os.path.join(save_directory, 'label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(save_directory, 'scaler.pkl'))
