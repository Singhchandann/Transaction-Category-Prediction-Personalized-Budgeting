from tokenizers import BertWordPieceTokenizer
import os

def train_tokenizer(texts, vocab_size=30522):
    tokenizer = BertWordPieceTokenizer(lowercase=True)
    tokenizer.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=2)
    return tokenizer

# Save the tokenizer
def save_tokenizer(tokenizer, save_directory='./custom_tokenizer'):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    tokenizer.save_model(save_directory)

# Load the custom tokenizer
def load_tokenizer(save_directory='./custom_tokenizer'):
    from transformers import BertTokenizerFast
    return BertTokenizerFast.from_pretrained(save_directory)
