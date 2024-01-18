import logging
import pandas as pd
import random
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from nltk.tokenize import sent_tokenize
import nltk
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import copy
import time
import unicodedata
import re
#logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Hello how are you doing?"
tokenized_text = tokenizer.tokenize(text)

masked_index = 2
tokenized_text[masked_index] = '[MASK]'
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokens_tensor = torch.tensor([indexed_tokens]).to(device)
segments_tensors = torch.tensor([segments_ids]).to(device)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to(device)
model.eval()



print(device)

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors).logits
    predictions = outputs

predicted_indexes = []
for i in range(6):
    predicted_indexes.append(torch.argmax(predictions[0, i]).item())
predicted_token = tokenizer.convert_ids_to_tokens(predicted_indexes)
print(predicted_token)