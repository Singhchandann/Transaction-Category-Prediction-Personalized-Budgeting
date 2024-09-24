import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the data from CSV
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess data
def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['Category'])

    scaler = MinMaxScaler()
    df['amount_normalized'] = scaler.fit_transform(df[['Amount']])

    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df['timestamp_numeric'] = df['Date'].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
   
    return df, label_encoder, scaler
