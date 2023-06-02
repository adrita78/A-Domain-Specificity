# A-Domain-Specificity

# BERT and LSTM Models for Sequence Classification
This repository contains both BERT (Bidirectional Encoder Representations from Transformers) and LSTM (Long Short-Term Memory) models implemented for sequence classification. These models are trained on a custom dataset and can be used to make predictions on new sequences.

## Installation

**Requirements**
- Python >= 3.6
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.4
- Transformers library (Hugging Face)

# Dataset
The dataset used for training, validation, and testing is located in the data directory. It consists of sequences and their corresponding labels in a tab-separated values (TSV) format. Each line in the TSV file represents a sequence-label pair.

## Model Architecture
**BERT Model**

The BERT model used for sequence classification consists of a pre-trained BERT base model followed by a classification layer. The pre-trained BERT base model is loaded from the bert-base-uncased checkpoint provided by the Transformers library. The classification layer is a fully connected layer that maps the BERT output to the number of classes.

**LSTM Model**

The LSTM model used for sequence classification consists of an embedding layer, an LSTM layer, and a classification layer. The embedding layer converts the input sequences into dense vectors. The LSTM layer processes the sequence data and captures sequential dependencies. The classification layer maps the LSTM output to the number of classes.

To run the AdomainDataset class, you can follow these steps:

Create an instance of the AdomainDataset class, providing the required parameters such as the filename, padding_token, and missing_token. For example:



