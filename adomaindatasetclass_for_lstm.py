# -*- coding: utf-8 -*-
"""AdomainDatasetClass for LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jmhOu7ITSsJBxiAq9I7hMLcMBHIvh72x
"""

class AdomainDataset(Dataset):
    def __init__(self, filename, padding_token='X', missing_token='-'):
        sequences, labels = self.load_file(filename)
        self.sequences = sequences
        self.labels = labels
        self.padding_token = padding_token
        self.missing_token = missing_token
        self.label_map = self.build_label_map(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return sequence, label

    def load_file(self, filename):
        sequences = []
        labels = []
        with open(filename, 'r') as file:
            for line in file:
                sequence, label = line.strip().split('\t')
                sequences.append(sequence)
                labels.append(label)
        return sequences, labels

    def collate_fn(self,batch):
      sequences, labels = zip(*batch)

    # Converting sequences to numerical representation (one-hot encoding)
      char_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
                     'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20, 'X':21}

    
      numerical_labels = [self.label_map[label] for label in labels]

    # Padding sequences to a fixed length and handle missing amino acids
      num_chars = len(char_to_index)
      encoded_sequences = [[char_to_index[char] if char in char_to_index else char_to_index[self.missing_token] for char in sequence] for sequence in sequences]
      padded_sequences = torch.nn.utils.rnn.pad_sequence([torch.tensor(encoded_sequence) for encoded_sequence in encoded_sequences], batch_first=True, padding_value=char_to_index[self.padding_token])
      padded_sequences = torch.nn.functional.one_hot(padded_sequences, num_classes=num_chars)

      numerical_labels = torch.tensor(numerical_labels)

      return padded_sequences, numerical_labels


    def build_label_map(self, labels):
      unique_labels = sorted(set(labels))
      label_map = {label: index for index, label in enumerate(unique_labels)}
      return label_map