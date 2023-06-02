# A-Domain-Specificity

# BERT and LSTM Models for Sequence Classification
This repository contains both BERT (Bidirectional Encoder Representations from Transformers) and LSTM (Long Short-Term Memory) models implemented for sequence classification. These models are trained on a custom dataset and can be used to make predictions on new sequences.

## Installation

**Requirements**
- Python >= 3.6
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.4
- Transformers library (Hugging Face)

Once you have installed the dependencies and downloaded the dataset, you can run the code by opening the Notebooks in your Google Colab environment.

# Dataset
The dataset used for training, validation, and testing is located in the data directory. It consists of sequences and their corresponding labels in a tab-separated values (TSV) format. Each line in the TSV file represents a sequence-label pair.

## Model Architecture
**BERT Model**

The BERT model used for sequence classification consists of a pre-trained BERT base model followed by a classification layer. The pre-trained BERT base model is loaded from the bert-base-uncased checkpoint provided by the Transformers library. The classification layer is a fully connected layer that maps the BERT output to the number of classes.

**LSTM Model**

The LSTM model used for sequence classification consists of an embedding layer, an LSTM layer, and a classification layer. The embedding layer converts the input sequences into dense vectors. The LSTM layer processes the sequence data and captures sequential dependencies. The classification layer maps the LSTM output to the number of classes.

To run the AdomainDataset class, you can follow these steps:

Create an instance of the AdomainDataset class, providing the required parameters such as the filename, padding_token, and missing_token. For example:

```angular2html
dataset = AdomainDataset(filename='data.txt', padding_token='X', missing_token='-')

```
Implement a data loader to iterate over the dataset. You can use the DataLoader class from PyTorch to achieve this. Import the necessary modules:

```angular2html
from torch.utils.data import DataLoader

```
Initialize a data loader object with the dataset and specify the batch size and any other desired parameters:


```angular2html
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

```

Iterate over the data loader to access the batches of data during training or evaluation. Each batch will contain a tuple of padded sequences and numerical labels:
```angular2html
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

```

The data file should be in the format where each line contains a sequence and its corresponding label separated by a tab character ('\t'). For example:

```angular2html
SEQUENCE1	LABEL1
SEQUENCE2	LABEL2
...
```
## Training Pipeline

The training pipeline consists of the following steps:

**Define Loss Function and Optimizer


-criterion: The loss function for training the model, which is the cross-entropy loss in this case.

-optimizer: The optimizer used to update the model parameters, in this case, Adam optimizer with a learning rate of 0.01.

## Training Loop

```angular2html
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

```















