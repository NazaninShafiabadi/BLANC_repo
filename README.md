# BLANC for Translation Quality Estimation: A Reimplementation and Adaptation of a Human-free Summary Quality Estimator

## Contributors 

Nazanin Shafiabadi, nazanin.shafiabadi@etu.u-paris.fr

Clara Gard, clara.gard94@gmail.com

Liora Taieb, liora.taieb@dauphine.eu

## Introduction

Translation quality estimation plays a crucial role in various applications, ensuring the accuracy and fluency of translated content. Our project focuses on adapting BLANC, a method originally designed for automatic document summary quality estimation, to the task of translation quality assessment. By leveraging a pre-trained language model, BLANC offers a human-free approach to evaluating translation quality, making it particularly suitable for low-resource languages and domains.

## Description

In this project, we propose a novel application of BLANC for translation quality estimation. While our method is language-agnostic, we have conducted evaluations primarily on English-to-French and English-to-Persian translation pairs. This choice was motivated by our familiarity with these languages, rather than any specific targeting of language pairs. BLANC evaluates the performance boost of a pre-trained language model when provided with a summary or translation of a document. Unlike traditional methods, BLANC does not rely on human references, enhancing its applicability across diverse linguistic contexts.

## Running the Program

To run the code, you need to create a new virtual environment and install the dependencies listed in the `requirements.txt` file. On Linux or Mac, you can do this with the following commands:

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Then, you can open the Jupyter notebooks and run the cells one by one to reproduce the `results.json` file.

## Limitations and Future Directions 

One of the limitations of our project is that we have not performed a comprehensive evaluation of our method. We have only tested our method on two language pairs, and we have not compared it with other metrics or human evaluations. These are essential aspects to measure the quality and usefulness of our method. As future directions, we aim to extend our experiments to more languages and domains, and to compare our method with other metrics such as BLEU and METEOR. We also intend to collect human feedback and correlate it with our method’s scores.

## Reference

```
@misc{vasilyev2020blanc,
      title         = {Fill in the BLANC: Human-free quality estimation of document summaries}, 
      author        = {Oleg Vasilyev and Vedant Dharnidharka and John Bohannon},
      year          = {2020},
      eprint        = {2002.09836},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CL}
      }
```
