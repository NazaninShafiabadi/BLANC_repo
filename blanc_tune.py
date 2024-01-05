import pandas as pd
import random as rd
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from torch.utils.data import DataLoader, Dataset
import torch

def blanc_tune(summary, text, model, p_mask = 0.15, l_min = 4, N = 10):
    words_in_summary = summary.split()
    N_summary = len(words_in_summary)
    N_mask = int(N_summary*p_mask)
    set_tune = pd.DataFrame(columns = ['summary', 'text']) # change if needed when importing true dataset
    for i in range(1, N + 1):
        pos = [i for i, word in enumerate(words_in_summary) if len(word) >= l_min]
        pos = rd.shuffle(pos)
        while len(pos) != 0:
            masked_summary = words_in_summary.copy()
            for pos_to_mask in pos[:N_mask]:
                masked_summary[pos_to_mask] = '[MASK]'
                set_tune.loc[set_tune.shape[0]] = [masked_summary, text]
    # add tuning of model (see below, from chatgpt, also look at homework 2)
            
    return

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

class ClozeDataset(Dataset):
    def __init__(self, texts, summaries):
        self.texts = texts
        self.summaries = summaries

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'summary': self.summaries[idx]}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

cloze_dataset = ClozeDataset(set_tune['text'], set_tune['summary'])
cloze_dataloader = DataLoader(cloze_dataset, batch_size=32, shuffle=True)

# Example training loop:
for epoch in range(3):  # Replace with the desired number of epochs
    model.train()
    for batch in cloze_dataloader:
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, return_special_tokens_mask=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        labels = tokenizer(batch['summary'], return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)

        # Ensure that labels are masked only at the [MASK] token positions
        labels[inputs['input_ids'] == tokenizer.mask_token_id] = -100

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Step 4: Save the fine-tuned model
model.save_pretrained('fine_tuned_bert_cloze_model')