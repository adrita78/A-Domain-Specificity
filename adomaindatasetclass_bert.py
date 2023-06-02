# -*- coding: utf-8 -*-
"""AdomainDatasetClass-BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EI5xv4E-NMmyABK4vbOL6HdPEsmkD6dF
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
        attention_mask = [1 if token != self.padding_token else 0 for token in sequence]
        return sequence, attention_mask, label

    def load_file(self, filename):
        sequences = []
        labels = []
        with open(filename, 'r') as file:
            for line in file:
                sequence, label = line.strip().split('\t')
                sequences.append(sequence)
                labels.append(label)
        return sequences, labels

    def collate_fn(self, batch):
      sequences, attention_masks, labels = zip(*batch)

    # Convert sequences to numerical representation (one-hot encoding)
      char_to_index = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
                     'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20, 'X': 21}

      numerical_labels = [self.label_map[label] for label in labels]

    # Find the maximum sequence length in the batch
      max_length = max(len(seq) for seq in sequences)

    # Pad sequences and attention masks to the maximum length
      padded_sequences = []
      padded_attention_masks = []
      for sequence, attention_mask in zip(sequences, attention_masks):
        padding_length = max_length - len(sequence)
        padded_sequence = [char_to_index[char] if char in char_to_index else char_to_index[self.missing_token] for char in sequence]
        padded_sequence += [char_to_index[self.padding_token]] * padding_length
        padded_sequences.append(padded_sequence)

        padded_attention_mask = attention_mask + [0] * padding_length
        padded_attention_masks.append(padded_attention_mask)

      numerical_labels = torch.tensor(numerical_labels)
      padded_sequences = torch.tensor(padded_sequences)
      padded_attention_masks = torch.tensor(padded_attention_masks)

      return padded_sequences, padded_attention_masks, numerical_labels




    def build_label_map(self, labels):
      unique_labels = sorted(set(labels))
      label_map = {label: index for index, label in enumerate(unique_labels)}
      return label_map