# GermanFinBERT

German FinBERT is a BERT language model focusing on the financial domain within the German language. In my [paper](https://arxiv.org/pdf/2311.08793.pdf), I describe in more detail the steps taken to train the model and show that it outperforms its generic benchmarks for finance specific downstream tasks. I train two versions of German FinBERT:
- Further pre-trained version (FP):
  This version of German FinBERT starts with the [gbert-base model](https://huggingface.co/deepset/gbert-base) and continues pre-training on finance specific textual data. The model can be found [here](https://huggingface.co/scherrmann/GermanFinBert_FP).
- Pre-trained from scratch version (SC):
  This version of German FinBERT is pre-trained from scratch on German finance specific textual data, starting with the Bert-base architecture and the vocabulary of the [bert-base-german-cased](https://huggingface.co/bert-base-german-cased) model of Deepset. The model can be found [here](https://huggingface.co/scherrmann/GermanFinBert_SC).

This repository does not contain the codes for pre-training the BERT models, as I use exactly the codes of [MosaicML](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert) for Huggingface models for this purpose. Therefore, I only share the fine-tuning  scripts.

  
## Overview

- Author: Moritz Scherrmann
- Paper: [here](https://arxiv.org/pdf/2311.08793.pdf)
- Architecture: BERT base
- Language: German
- Specialization: Financial textual data
- Original Model: [gbert-base model](https://huggingface.co/deepset/gbert-base) (deepset)
- Framework: [MosaicML](https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert)
  
## Pre-training

German FinBERT's pre-training corpus includes a diverse range of financial documents, such as Bundesanzeiger reports, Handelsblatt articles, MarketScreener data, and additional sources including FAZ, ad-hoc announcements, LexisNexis & Event Registry content, Zeit Online articles, Wikipedia entries, and Gabler Wirtschaftslexikon. In total, the corpus spans from 1996 to 2023, consisting of 12.15 million documents with 10.12 billion tokens over 53.19 GB.

I use the following pre-training set-ups:
- Further pre-training:
  I further pre-train the model for 10,400 steps with a batch size of 4096, which is one epoch. I use an Adam optimizer with decoupled weight decay regularization, with Adam parameters 0.9, 0.98, 1e − 6, a weight decay of 1e − 5 and a maximal learning of 1e − 4. I train the model using a Nvidia DGX A100 node consisting of 8 A100 GPUs with 80 GB of memory each.
- Pre-training from scratch:
  With a batch size of 4096, I train the German FinBERT model for 174,000 steps, summing up to more than 17 epochs. I use an Adam optimizer with decoupled weight decay regularization, with Adam parameters 0.9, 0.98, 1e − 6, a weight decay of 1e − 5 and a maximal learning of 5e − 4. I train the model using a Nvidia DGX A100 node consisting of 8 A100 GPUs with 80 GB of memory each.  

## Performance
### Fine-tune Datasets

To fine-tune the model, I use several datasets, including:
- A manually labeled multi-label database of German ad-hoc announcements containing 31,771 sentences, each associated with up to 20 possible topics.
- An extractive question-answering dataset based on the SQuAD format, which was created using 3,044 ad-hoc announcements processed by OpenAI's ChatGPT to generate and answer questions (see [here]()).
- The financial phrase bank of Malo et al. (2013) for sentiment classification, translated to German using DeepL (see [here](https://huggingface.co/datasets/scherrmann/financial_phrasebank_75agree_german)).

### Benchmark Results

The further pre-trained German FinBERT model demonstrated the following performances on finance-specific downstream tasks:

Ad-Hoc Multi-Label Database:
- Macro F1: 86.08%
- Micro F1: 85.65%

Ad-Hoc QuAD (Question Answering):
- Exact Match (EM): 52.50%
- F1 Score: 74.61%

Translated Financial Phrase Bank:
- Accuracy: 95.41%
- Macro F1: 91.49%

The further pre-trained German FinBERT model demonstrated the following performances on finance-specific downstream tasks:

Ad-Hoc Multi-Label Database:
- Macro F1: 85.67%
- Micro F1: 85.17%

Ad-Hoc QuAD (Question Answering):
- Exact Match (EM): 50.23%
- F1 Score: 72.80%

Translated Financial Phrase Bank:
- Accuracy: 95.95%
- Macro F1: 92.70%

## Author

Moritz Scherrmann: scherrmann [at] lmu.de

For additional details regarding the performance on fine-tune datasets and benchmark results, please refer to the full documentation provided in the study.
