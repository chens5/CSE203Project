import torch
import numpy as np

def convert_one_hot_to_categorical(instance):
    return np.argmax(instance, axis=-1)

# data/3 contains 9x9 sudoku grids
# data/2 contains 4x4 sudoku grids
# all features and labels are one-hot encoded on the values in the grid
# so we get batch_size x k x k x k tensors, based on how the data
# was constructed the final 'k' axis in the tensor is the one-hot encoding

features_filepath = 'sudoku/data/2/features.pt'
labels_filepath = 'sudoku/data/2/labels.pt'

features = torch.load(features_filepath)
labels = torch.load(labels_filepath)

print(f'data/2/features.pt shape: {features.size()}')
print(f'data/2/labels.pt shape: {labels.size()}')

features_filepath = 'sudoku/data/3/features.pt'
labels_filepath = 'sudoku/data/3/labels.pt'

features = torch.load(features_filepath)
labels = torch.load(labels_filepath)

print(f'data/3/features.pt shape: {features.size()}')
print(f'data/3/labels.pt shape: {labels.size()}')

incomplete_sample = convert_one_hot_to_categorical(features[0])
complete_sample = convert_one_hot_to_categorical(labels[0])

print('incomplete sample:')
print(incomplete_sample)

print('complete sample:')
print(complete_sample)
