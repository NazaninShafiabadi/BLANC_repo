from datasets import load_dataset
import pandas as pd

grader = "Clara"

en_fr_ds = load_dataset('news_commentary', 'en-fr', split='train')

en_fr_ds = pd.DataFrame(en_fr_ds['translation'])

en_fr_ds = en_fr_ds[en_fr_ds["en"].str.split().str.len() > 10]
en_fr_ds = en_fr_ds[en_fr_ds["fr"].str.split().str.len() > 10]

en_fr_ds = en_fr_ds[:3]

en_fr_ds[grader] = -1


for index, row in en_fr_ds.iterrows():
    print(index, row)
    print("sentence: ", row["fr"])
    print("translation:", row["en"])
    print("input grade: ")
    en_fr_ds[grader][index] = int(input())


en_fr_ds.to_csv("en-fr_translations.csv")