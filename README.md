# Transaction-Category-Prediction-Personalized-Budgeting-BERT-Model
Welcome to the Transaction Category Prediction project! This project uses BERT to categorize financial transactions based on subcategory, notes, and amount, and provides personalized budget suggestions and investment strategies based on your input.  
## Key Features:  

Custom Tokenizer: Train a custom tokenizer from transaction data.
BERT-Based Classification: Fine-tune a BERT model to predict transaction categories.  
Personalized Budgeting: Generate budget suggestions based on categorized spending.  
Investment Strategies: Recommend investments based on user risk tolerance and financial profile.    
Dataset:
We use the Daily Household Transactions Dataset, containing records of financial transactions including:   

Subcategory (e.g., groceries, utilities)
Amount spent
Notes providing additional transaction context
The dataset is available in the data/ directory.  

Installation:
To get started with this project, clone the repository and install the required libraries:

```bash
git clone https://github.com/Singhchandann/Transaction-Category-Prediction-Personalized-Budgeting.git
```
```bash
cd category-prediction-model
```
```bash
pip install -r requirements.txt
```

Steps to Run the Project:
1. Load and Preprocess Data
We preprocess the dataset, encoding categorical labels and normalizing transaction amounts for optimal model training.

```python
from scripts.data_loader import load_data, preprocess_data

file_path = 'data/Daily Household Transactions.csv'
df, label_encoder, scaler = preprocess_data(load_data(file_path))
```  

2. Train the Custom Tokenizer
We use BertWordPieceTokenizer to tokenize the transaction data:

```python
from scripts.tokenizer_trainer import train_tokenizer

texts = df.apply(lambda row: f"Subcategory: {row['Subcategory']} Note: {row['Note']} Amount: {row['Amount']}", axis=1).tolist()
tokenizer = train_tokenizer(texts, vocab_size=30522)
tokenizer.save_model('models/custom_tokenizer')
```

3. Fine-Tune BERT Model
Train a BERT-based model for sequence classification:

```bash
python scripts/model_training.py
```
This script fine-tunes the pre-trained BERT model for transaction category classification.  

4. Category Prediction & Budget Suggestions
Run the model to predict transaction categories and suggest personalized budgets:

```bash
python scripts/budgeting_and_investment.py
```  
You will be prompted to input your age, income, risk tolerance, and log transactions. The model will categorize the transactions, suggest budgets, and recommend investment strategies.
  
Example Usage:
```python
from scripts.model_inference import predict_category

category = predict_category(model, tokenizer, 'Groceries', 'Bought vegetables', 150.0)
print(f'Transaction categorized as: {category}')
```

Results:
Budget Suggestions
After analyzing the logged transactions, the system provides budget recommendations:

```text
Category  | Suggested Budget
----------------------------
Groceries | $400.00
Utilities | $100.00
Transport | $50.00
...
```

Investment Recommendations
Based on user profile and remaining balance, investment recommendations are provided, such as:

```text
Investment Strategies:
- Stocks: $300.00
- Real Estate: $200.00
- Emergency Fund: $100.00
...
```

### Files Explanation:
data_loader.py: Contains functions for loading and preprocessing the data.  

tokenizer_trainer.py: Trains a custom tokenizer from transaction data.  

model_training.py: Fine-tunes a BERT model for transaction category prediction.  

model_inference.py: Includes functions for making category predictions from new inputs.  

budgeting_and_investment.py: Contains logic for personalized budget and investment suggestions based on user input and transactions.  

License:
This project is licensed under the MIT License - see the LICENSE file for details.

### Future Enhancements:
More Advanced NLP: Explore incorporating additional transaction context into category prediction.  

Recurrent Models: Experiment with RNNs or transformers for time-series analysis of monthly spending.  

Financial Risk Models: Add more granular risk profiling for investment strategies.  
