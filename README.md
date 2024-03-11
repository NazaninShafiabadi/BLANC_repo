# BLANC for Translation Quality Estimation: A Reimplementation and Adaptation of a Human-free Summary Quality Estimator

## Contributors 

Nazanin Shafiabadi, nazanin.shafiabadi@etu.u-paris.fr

Clara Gard, clara.gard94@gmail.com

Liora Taieb, liora.taieb@dauphine.eu

## Introduction

Translation quality estimation plays a crucial role in various applications, ensuring the accuracy and fluency of translated content. Our project focuses on adapting BLANC, a method originally designed for automatic document summary quality estimation, to the task of translation quality assessment. By leveraging a pre-trained language model, BLANC offers a human-free approach to evaluating translation quality, making it particularly suitable for low-resource languages and domains.

## Description

In this project, we propose a novel application of BLANC for translation quality estimation. While our method is language-agnostic, we have conducted evaluations primarily on English-to-French and English-to-Persian translation pairs. This choice was motivated by our familiarity with these languages, rather than any specific targeting of language pairs. BLANC evaluates the performance boost of a pre-trained language model carrying out its language understanding task on a document, when provided with a summary or translation of the document. Unlike traditional methods, BLANC does not rely on human references, enhancing its applicability across diverse linguistic contexts.

## Running the Program

To run the program, you need to have [Pipenv] installed on your system. Pipenv is a tool that manages Python dependencies and virtual environments.

Follow these steps to run the program:

1. Clone this repository to your local machine.
2. Navigate to the project directory and run `pipenv shell` to create and activate a virtual environment.
3. Run `pipenv install` to install all the dependencies from the Pipfile.
4. Open the Jupyter notebooks and execute the cells sequentially to reproduce the `results.json` file.

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
