import torch
import nltk
import asyncio
from concurrent.futures import ThreadPoolExecutor
import regex as re
import unicodedata
import json

nltk.download('punkt')


def preprocess_text(text: str) -> str:
     # lower case
     text = text.lower()

     # before normalization : manual handling of contractions and line breaks
     text = text.replace('\n', ' ')
     text = text.replace(' \' ', '\'')
     text = text.replace('\'', '')

     # string normalization.
     text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
     text = str(text)[2:-1]
     # the result of previous line adds a few characters to the string,
     # we remove them.

     # remove non alpha numeric characters, except dots, question and exclamation marks that will be needed to separate sentences.
     text = re.sub(r'[^\w]', ' ', text)

     # replace numbers by the <NUM> token.
     text = re.sub(r'[0-9]+', '<NUM>', text)

     # remove double whitespaces.
     text = re.sub(r'( ){2,}', ' ', text).strip()
     # removing spaces at beginning and end of string.

     return text


def mask_sentence(sentence, mask_token, i, M, L_min):
    return [mask_token
            if (j - i) % M == 0
            and (len(sentence[j]) >= L_min
                 or sentence[j].startswith('##')
                 or sentence[min(j+1, len(sentence)-1)].startswith('##'))
            else sentence[j]
            for j in range(len(sentence))]


def no_copy_guard(sentence, summary):
    sentence = ' '.join(sentence)
    summary = ' '.join(summary)
    return sentence in summary


def BLANC_help_summary(text, summary, model, tokenizer, M=6, L_min=4, sep='[SEP]', device='cpu', word_sim_model = None):
    """
    Calculates BLANC score between a given text and its summary using a specified model.

    Parameters:
    - text (List[List[str]]): List of sentences represented as a list of tokens.
    - summary (List[str]): The tokenized summary of the text.
    - model: BERT-type model
    - tokenizer: The tokenizer associated with the model used.
    - M (int): Parameter M for the algorithm (default is 6).
    - L_min (int): Minimum length requirement for masked words (default is 4).
    - sep (str): Separator between the inference help (filler/summary) and a sentence from the text (default is '[SEP]').

    Returns:
    - float: BLANC score for the given text and its summary.
    """

    filler = ['.'] * len(summary)
    S = [[0, 0], [0, 0]]

    score = torch.zeros(1).to(device)[0]

    for sentence in text:

        if no_copy_guard(sentence, summary): 
           continue

        for i in range(M):
            masked_sentence = mask_sentence(sentence, tokenizer.mask_token, i, M, L_min)

            input_base = filler + [sep] + masked_sentence
            input_help = summary + [sep] + masked_sentence

            tokenized_input_base = torch.tensor(tokenizer.convert_tokens_to_ids(input_base)).to(device) # Shape: [sequence_length]
            tokenized_input_help = torch.tensor(tokenizer.convert_tokens_to_ids(input_help)).to(device) # Shape: [sequence_length]
            
            with torch.no_grad():
                input_stacked = torch.stack((tokenized_input_base, tokenized_input_help))
                out_stacked = model(input_ids=input_stacked).logits  # Shape: [1, sequence_length, Bert_vocab_size]
                out_base = out_stacked[0]
                out_help = out_stacked[1]

            out_base = torch.argmax(out_base.squeeze(0), dim=-1)  # Shape: [sequence_length]
            out_help = torch.argmax(out_help.squeeze(0), dim=-1)  # Shape: [sequence_length]

            masked_tokens = [idx for idx, word in enumerate(masked_sentence) if word == tokenizer.mask_token]

            for j in masked_tokens:
                idx = len(summary + [sep]) + j
                predicted_word_base = tokenizer.convert_ids_to_tokens(out_base[idx].item())
                predicted_word_help = tokenizer.convert_ids_to_tokens(out_help[idx].item())

                if word_sim_model is not None:
                    predicted_sentence_base = tokenizer.convert_tokens_to_ids(masked_sentence)
                    predicted_sentence_base[j] = out_base[idx].item()

                    predicted_sentence_help = tokenizer.convert_tokens_to_ids(masked_sentence)
                    predicted_sentence_help[j] = out_help[idx].item()

                    tokenized_sentence = masked_sentence.copy()
                    tokenized_sentence[j] = sentence[j]
                    tokenized_sentence = tokenizer.convert_tokens_to_ids(tokenized_sentence)

                    with torch.no_grad():
                        word_sim_input = torch.stack([torch.tensor(predicted_sentence_base), torch.tensor(predicted_sentence_help), torch.tensor(tokenized_sentence)]).to(device)
                        word_sim_out = word_sim_model(word_sim_input)
                        predicted_base_embedding = word_sim_out.last_hidden_state[0, j, :]
                        predicted_help_embedding = word_sim_out.last_hidden_state[1, j, :]
                        correct_embedding = word_sim_out.last_hidden_state[2, j, :]

                    cos_sim = torch.nn.CosineSimilarity(dim=0)
                    
                    sim_base = cos_sim(predicted_base_embedding, correct_embedding)
                    sim_help = cos_sim(predicted_help_embedding, correct_embedding)

                    k = int(sim_base > 0.98)
                    m = int(sim_help > 0.98)
                    S[k][m] += 1

                    score += (sim_help - sim_base)/(sim_base)

                else:
                    k = int(predicted_word_base == sentence[j])
                    m = int(predicted_word_help == sentence[j])
                    S[k][m] += 1


    B = (S[0][1] - S[1][0]) / (S[0][0] + S[1][1] + S[0][1] + S[1][0])

    return B, score.item() / (S[0][0] + S[1][1] + S[0][1] + S[1][0])


def BLANC_help_translation(sentence, translation, model, tokenizer, M=6, L_min=4, sep='[SEP]', device='cpu'):
    """
    Calculates BLANC score between a given sentence and its translation using a specified model.

    Parameters:
    - sentence (List[str]): A tokenized sentence.
    - translation (List[str]): The tokenized translation.
    - model: BERT-type model
    - tokenizer: The tokenizer associated with the model used.
    - M (int): Parameter M for the algorithm (default is 6).
    - L_min (int): Minimum length requirement for masked words (default is 4).
    - sep (str): Separator between the inference help (filler/summary) and a sentence from the text (default is '[SEP]').

    Returns:
    - float: BLANC score for the given sentence and its translation.
    """
    filler = ['.'] * len(translation)
    S = [[0, 0], [0, 0]]

    for i in range(M):
        masked_sentence = mask_sentence(sentence, tokenizer.mask_token, i, M, L_min)

        input_base = filler + [sep] + masked_sentence
        input_help = translation + [sep] + masked_sentence

        tokenized_input_base = torch.tensor(tokenizer.convert_tokens_to_ids(input_base)).to(device) # Shape: [sequence_length]
        tokenized_input_help = torch.tensor(tokenizer.convert_tokens_to_ids(input_help)).to(device) # Shape: [sequence_length]

        out_base = model(input_ids=tokenized_input_base.unsqueeze(0)).logits  # Shape: [1, sequence_length, model_vocab_size]
        out_help = model(input_ids=tokenized_input_help.unsqueeze(0)).logits  # Shape: [1, sequence_length, model_vocab_size]

        out_base = torch.argmax(out_base.squeeze(0), dim=-1)  # Shape: [sequence_length]
        out_help = torch.argmax(out_help.squeeze(0), dim=-1)  # Shape: [sequence_length]

        masked_tokens = [idx for idx, word in enumerate(masked_sentence) if word == tokenizer.mask_token]

        for j in masked_tokens:
            idx = len(translation + [sep]) + j
            predicted_word_base = tokenizer.convert_ids_to_tokens(out_base[idx].item())
            predicted_word_help = tokenizer.convert_ids_to_tokens(out_help[idx].item())

            k = int(predicted_word_base == sentence[j])
            m = int(predicted_word_help == sentence[j])
            S[k][m] += 1

    if (S[0][0] + S[1][1] + S[0][1] + S[1][0]) == 0:
        B = 0
    else:
        B = (S[0][1] - S[1][0]) / (S[0][0] + S[1][1] + S[0][1] + S[1][0])

    return B


def add_results_to_json(new_data, file_path = "./results.json"):
    try:
        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = {}

    for key, value in new_data.items():
        existing_data[key] = value

    with open(file_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=2)

    print(f"Data has been added to {file_path}")


def study_results(dataset, scores, score_lower_bound=-1, score_upper_bound=1, verbose=False):
    num_examples = 0
    num_scores = dataset.shape[0]
    for idx, score in enumerate(scores[:num_scores]):
        if score_lower_bound <= score < score_upper_bound:
            num_examples += 1
            if dataset.shape[1] == 3:
                print(f'Example {idx}   score: {score}   annotator score: {dataset.iloc[idx, 2]}')
            else:
                print(f'Example {idx}   score: {score}   annotator score: -')
            if verbose:
                print(f'Sentence: {dataset.iloc[idx, 0]}')
                print(f'Translation: {dataset.iloc[idx, 1]}')
            print('-' * 100)
    print(f'{num_examples}/{dataset.shape[0]} scores were between {score_lower_bound} and {score_upper_bound}.')