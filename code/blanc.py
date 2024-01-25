from datasets import Dataset
import torch
import nltk
import regex as re
import unicodedata
import json
import random

from transformers import BertForMaskedLM, TrainingArguments, Trainer

nltk.download("punkt")

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids = torch.tensor([example["input_ids"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )  # -100 is the default index to ignore in the loss function of the Trainer

        return {"input_ids": input_ids, "labels": labels}


def preprocess_text(text: str) -> str:
    # lower case
    text = text.lower()

    # before normalization : manual handling of contractions and line breaks
    text = text.replace("\n", " ")
    text = text.replace(" ' ", "'")
    text = text.replace("'", "")

    # string normalization.
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore")
    text = str(text)[2:-1]
    # the result of previous line adds a few characters to the string,
    # we remove them.

    # remove non alpha numeric characters, except dots, question and exclamation marks that will be needed to separate sentences.
    text = re.sub(r"[^\w]", " ", text)

    # replace numbers by the <NUM> token.
    text = re.sub(r"[0-9]+", "<NUM>", text)

    # remove double whitespaces.
    text = re.sub(r"( ){2,}", " ", text).strip()
    # removing spaces at beginning and end of string.

    return text


def tune_model(tune_set, model, tokenizer, n_epochs):
    training_args = TrainingArguments(
        f"finetuned-model",
        evaluation_strategy="no",
        learning_rate=1e-4,
        weight_decay=0.01,
        num_train_epochs=n_epochs,
    )

    data_collator = CustomDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tune_set,
        data_collator=data_collator,
    )

    trainer.train()

    return model


def mask_sentence(sentence, mask_token, i, M, L_min):
    return [
        mask_token
        if (j - i) % M == 0
        and (
            len(sentence[j]) >= L_min
            or sentence[j].startswith("##")
            or sentence[min(j + 1, len(sentence) - 1)].startswith("##")
        )
        else sentence[j]
        for j in range(len(sentence))
    ]


def no_copy_guard(sentence, summary):
    sentence = " ".join(sentence)
    summary = " ".join(summary)
    return sentence in summary


def BLANC_help_summary(
    text,
    summary,
    model,
    tokenizer,
    M=6,
    L_min=4,
    sep="[SEP]",
    device="cpu",
    word_sim_model=None,
):
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

    filler = ["."] * len(summary)
    S = [[0, 0], [0, 0]]

    score = torch.zeros(1).to(device)[0]

    for sentence in text:
        if no_copy_guard(sentence, summary):
            continue

        for i in range(M):
            masked_sentence = mask_sentence(sentence, tokenizer.mask_token, i, M, L_min)

            input_base = filler + [sep] + masked_sentence
            input_help = summary + [sep] + masked_sentence

            tokenized_input_base = torch.tensor(
                tokenizer.convert_tokens_to_ids(input_base)
            ).to(
                device
            )  # Shape: [sequence_length]
            tokenized_input_help = torch.tensor(
                tokenizer.convert_tokens_to_ids(input_help)
            ).to(
                device
            )  # Shape: [sequence_length]

            with torch.no_grad():
                input_stacked = torch.stack(
                    (tokenized_input_base, tokenized_input_help)
                )
                out_stacked = model(
                    input_ids=input_stacked
                ).logits  # Shape: [1, sequence_length, Bert_vocab_size]
                out_base = out_stacked[0]
                out_help = out_stacked[1]

            out_base = torch.argmax(
                out_base.squeeze(0), dim=-1
            )  # Shape: [sequence_length]
            out_help = torch.argmax(
                out_help.squeeze(0), dim=-1
            )  # Shape: [sequence_length]

            masked_tokens = [
                idx
                for idx, word in enumerate(masked_sentence)
                if word == tokenizer.mask_token
            ]

            for j in masked_tokens:
                idx = len(summary + [sep]) + j
                predicted_word_base = tokenizer.convert_ids_to_tokens(
                    out_base[idx].item()
                )
                predicted_word_help = tokenizer.convert_ids_to_tokens(
                    out_help[idx].item()
                )

                if word_sim_model is not None:
                    predicted_sentence_base = tokenizer.convert_tokens_to_ids(
                        masked_sentence
                    )
                    predicted_sentence_base[j] = out_base[idx].item()

                    predicted_sentence_help = tokenizer.convert_tokens_to_ids(
                        masked_sentence
                    )
                    predicted_sentence_help[j] = out_help[idx].item()

                    tokenized_sentence = masked_sentence.copy()
                    tokenized_sentence[j] = sentence[j]
                    tokenized_sentence = tokenizer.convert_tokens_to_ids(
                        tokenized_sentence
                    )

                    with torch.no_grad():
                        word_sim_input = torch.stack(
                            [
                                torch.tensor(predicted_sentence_base),
                                torch.tensor(predicted_sentence_help),
                                torch.tensor(tokenized_sentence),
                            ]
                        ).to(device)
                        word_sim_out = word_sim_model(word_sim_input)
                        predicted_base_embedding = word_sim_out.last_hidden_state[
                            0, j, :
                        ]
                        predicted_help_embedding = word_sim_out.last_hidden_state[
                            1, j, :
                        ]
                        correct_embedding = word_sim_out.last_hidden_state[2, j, :]

                    cos_sim = torch.nn.CosineSimilarity(dim=0)

                    sim_base = cos_sim(predicted_base_embedding, correct_embedding)
                    sim_help = cos_sim(predicted_help_embedding, correct_embedding)

                    k = int(sim_base > 0.98)
                    m = int(sim_help > 0.98)
                    S[k][m] += 1

                    score += (sim_help - sim_base) / (sim_base)

                else:
                    k = int(predicted_word_base == sentence[j])
                    m = int(predicted_word_help == sentence[j])
                    S[k][m] += 1

    B = (S[0][1] - S[1][0]) / (S[0][0] + S[1][1] + S[0][1] + S[1][0])

    return B, score.item() / (S[0][0] + S[1][1] + S[0][1] + S[1][0])


def BLANC_help_translation(
    sentence, translation, model, tokenizer, M=6, L_min=4, sep="[SEP]", device="cpu"
):
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
    filler = ["."] * len(translation)
    S = [[0, 0], [0, 0]]

    for i in range(M):
        masked_sentence = mask_sentence(sentence, tokenizer.mask_token, i, M, L_min)

        input_base = filler + [sep] + masked_sentence
        input_help = translation + [sep] + masked_sentence

        tokenized_input_base = torch.tensor(
            tokenizer.convert_tokens_to_ids(input_base)
        ).to(
            device
        )  # Shape: [sequence_length]
        tokenized_input_help = torch.tensor(
            tokenizer.convert_tokens_to_ids(input_help)
        ).to(
            device
        )  # Shape: [sequence_length]

        out_base = model(
            input_ids=tokenized_input_base.unsqueeze(0)
        ).logits  # Shape: [1, sequence_length, model_vocab_size]
        out_help = model(
            input_ids=tokenized_input_help.unsqueeze(0)
        ).logits  # Shape: [1, sequence_length, model_vocab_size]

        out_base = torch.argmax(out_base.squeeze(0), dim=-1)  # Shape: [sequence_length]
        out_help = torch.argmax(out_help.squeeze(0), dim=-1)  # Shape: [sequence_length]

        masked_tokens = [
            idx
            for idx, word in enumerate(masked_sentence)
            if word == tokenizer.mask_token
        ]

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


def BLANC_tune_summary_inference(
    text, model, model_tuned, tokenizer, p_mask=0.15, L_min=4, device="cpu"
):
    """
    Compares the performance of a model fine-tuned on the 'summary' vs. a model that has never seen the summary.

    Parameters:
    - text (List[List[str]]): List of sentences represented as a list of tokens.
    - model: BERT-type model
    - model_tuned: The fine-tuned model.
    - tokenizer: The tokenizer associated with the model used.
    - p_mask (float): Probability of masking (default is 0.15).
    - L_min (int): Minimum length requirement for masked words (default is 4).

    Returns:
    - float: BLANC_tune score showing the quality of the summary.
    """

    S = [[0, 0], [0, 0]]
    M = int(1 / p_mask)

    for sentence in text:
        for i in range(M):
            masked_sentence = mask_sentence(sentence, tokenizer.mask_token, i, M, L_min)
            masked_sentence_ids = torch.tensor(
                tokenizer.convert_tokens_to_ids(masked_sentence)
            ).to(
                device
            )  # Shape: [sequence_length]

            out_base = model(
                input_ids=masked_sentence_ids.unsqueeze(0)
            ).logits  # Shape: [1, sequence_length, Bert_vocab_size]
            out_tune = model_tuned(
                input_ids=masked_sentence_ids.unsqueeze(0)
            ).logits  # Shape: [1, sequence_length, Bert_vocab_size]

            out_base = torch.argmax(
                out_base.squeeze(0), dim=-1
            )  # Shape: [sequence_length]
            out_tune = torch.argmax(
                out_tune.squeeze(0), dim=-1
            )  # Shape: [sequence_length]

            masked_tokens = [
                idx
                for idx, word in enumerate(masked_sentence)
                if word == tokenizer.mask_token
            ]

            for j in masked_tokens:
                predicted_word_base = tokenizer.convert_ids_to_tokens(
                    out_base[j].item()
                )
                predicted_word_tune = tokenizer.convert_ids_to_tokens(
                    out_tune[j].item()
                )

                k = int(predicted_word_base == sentence[j])
                m = int(predicted_word_tune == sentence[j])
                S[k][m] += 1

    B = (S[0][1] - S[1][0]) / (S[0][0] + S[1][1] + S[0][1] + S[1][0])

    return B


def BLANC_tune_translation_inference(
    sentence, model, model_tuned, tokenizer, p_mask=0.15, L_min=4, device="cpu"
):
    """
    Compares the performance of a model fine-tuned on the 'translation' vs. a model that has never seen the translation.

    Parameters:
    - sentence (List[str]): A tokenized sentence.
    - model: BERT-type model
    - model_tuned: The fine-tuned model.
    - tokenizer: The tokenizer associated with the model used.
    - p_mask (float): Probability of masking (default is 0.15).
    - L_min (int): Minimum length requirement for masked words (default is 4).

    Returns:
    - float: BLANC_tune score showing the quality of the translation.
    """

    S = [[0, 0], [0, 0]]
    M = int(1 / p_mask)

    for i in range(M):
        masked_sentence = mask_sentence(sentence, tokenizer.mask_token, i, M, L_min)
        masked_sentence_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(masked_sentence)
        ).to(
            device
        )  # Shape: [sequence_length]

        out_base = model(
            input_ids=masked_sentence_ids.unsqueeze(0)
        ).logits  # Shape: [1, sequence_length, Bert_vocab_size]
        out_tune = model_tuned(
            input_ids=masked_sentence_ids.unsqueeze(0)
        ).logits  # Shape: [1, sequence_length, Bert_vocab_size]

        out_base = torch.argmax(out_base.squeeze(0), dim=-1)  # Shape: [sequence_length]
        out_tune = torch.argmax(out_tune.squeeze(0), dim=-1)  # Shape: [sequence_length]

        masked_tokens = [
            idx
            for idx, word in enumerate(masked_sentence)
            if word == tokenizer.mask_token
        ]

        for j in masked_tokens:
            predicted_word_base = tokenizer.convert_ids_to_tokens(out_base[j].item())
            predicted_word_tune = tokenizer.convert_ids_to_tokens(out_tune[j].item())

            k = int(predicted_word_base == sentence[j])
            m = int(predicted_word_tune == sentence[j])
            S[k][m] += 1

    B = (S[0][1] - S[1][0]) / (S[0][0] + S[1][1] + S[0][1] + S[1][0])

    return B


def BLANC_tune_summary(
    text,
    summary,
    model_checkpoint,
    model,
    tokenizer,
    p_mask=0.15,
    L_min=4,
    N=10,
    n_epochs=3,
    device="cpu",
):
    # Model tuning
    N_words = len(summary)
    N_mask = int(N_words * p_mask)
    set_tune = Dataset.from_dict({})

    summary_ids = tokenizer.convert_tokens_to_ids(summary)

    for _ in range(N):
        pos = [
            i
            for i, token in enumerate(summary)
            if (
                len(token) >= L_min
                or token.startswith("##")
                or summary[min(i + 1, len(summary) - 1)].startswith("##")
            )
        ]  # positions of words longer than Lmin
        random.shuffle(pos)
        while len(pos) != 0:
            # Mask words in next N_mask positions
            masked_summary = summary_ids.copy()
            for pos_to_mask in pos[:N_mask]:
                masked_summary[pos_to_mask] = tokenizer.mask_token_id
            # Add translation with masked words to set_tune
            set_tune = set_tune.add_item(
                {"input_ids": masked_summary, "labels": summary_ids}
            )
            pos = pos[N_mask:]

    # Creating a fresh pre-trained model
    if len(set_tune) > 0:
        new_model = BertForMaskedLM.from_pretrained(model_checkpoint).to(device)
        model_tuned = tune_model(set_tune, new_model, tokenizer, n_epochs)

        # Comparing inference with model vs. model_tuned
        score = BLANC_tune_summary_inference(text, model, model_tuned, tokenizer, p_mask, L_min, device)

        del new_model
        del model_tuned
        torch.cuda.empty_cache()
        
    else:
        score = 0.0

    return score


def BLANC_tune_translation(
    sentence,
    translation,
    model_checkpoint,
    model,
    tokenizer,
    p_mask=0.15,
    L_min=4,
    N=10,
    n_epochs=3,
    device="cpu",
):
    # Model tuning
    N_words = len(translation)
    N_mask = int(N_words * p_mask)
    set_tune = Dataset.from_dict({})

    translation_ids = tokenizer.convert_tokens_to_ids(translation)

    for _ in range(N):
        pos = [
            i
            for i, token in enumerate(translation)
            if (
                len(token) >= L_min
                or token.startswith("##")
                or translation[min(i + 1, len(translation) - 1)].startswith("##")
            )
        ]  # positions of words longer than Lmin
        random.shuffle(pos)
        while len(pos) != 0:
            # Mask words in next N_mask positions
            masked_translation = translation_ids.copy()
            for pos_to_mask in pos[:N_mask]:
                masked_translation[pos_to_mask] = tokenizer.mask_token_id
            # Add translation with masked words to set_tune
            set_tune = set_tune.add_item(
                {"input_ids": masked_translation, "labels": translation_ids}
            )
            pos = pos[N_mask:]

    # Creating a fresh pre-trained model
    if len(set_tune) > 0:
        new_model = BertForMaskedLM.from_pretrained(model_checkpoint).to(device)
        model_tuned = tune_model(set_tune, new_model, tokenizer, n_epochs)

        # Comparing inference with model vs. model_tuned
        score = BLANC_tune_translation_inference(sentence, model, model_tuned, tokenizer, p_mask, L_min, device)

        del new_model
        del model_tuned
        torch.cuda.empty_cache()
        
    else:
        score = 0.0

    return score


def add_results_to_json(new_data, file_path="./results.json"):
    try:
        with open(file_path, "r") as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = {}

    for key, value in new_data.items():
        existing_data[key] = value

    with open(file_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=2)

    print(f"Data has been added to {file_path}")


def study_results(
    dataset, scores, score_lower_bound=-1, score_upper_bound=1, verbose=False
):
    num_examples = 0
    num_scores = dataset.shape[0]
    for idx, score in enumerate(scores[:num_scores]):
        if score_lower_bound <= score < score_upper_bound:
            num_examples += 1
            if dataset.shape[1] == 3:
                print(
                    f"Example {idx}   score: {score}   annotator score: {dataset.iloc[idx, 2]}"
                )
            else:
                print(f"Example {idx}   score: {score}   annotator score: -")
            if verbose:
                print(f"Sentence: {dataset.iloc[idx, 0]}")
                print(f"Translation: {dataset.iloc[idx, 1]}")
            print("-" * 100)
    print(
        f"{num_examples}/{dataset.shape[0]} scores were between {score_lower_bound} and {score_upper_bound}."
    )
