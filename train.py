import torch
import numpy as np
from models.optnet_cvxpylayers import OptNet

torch.manual_seed(309)
np.random.seed(484)

def build_dataloaders(features_path, labels_path, dev_percent=0.2):
    with open(features_path, 'rb') as fp:
        features = torch.load(fp).double()
    with open(labels_path, 'rb') as fp:
        labels = torch.load(fp).double()

    n_instances, n_features = features.size(0), int(np.prod(features.size()[1:]))
    n_train = int(n_instances*(1.-dev_percent))
    n_test = n_instances-n_train

    train_tensors = [features[:n_train], labels[:n_train]]
    dev_tensors = [features[n_train:], labels[n_train:]]

    train_dataset = torch.utils.data.TensorDataset(*train_tensors)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    dev_dataset = torch.utils.data.TensorDataset(*dev_tensors)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=0)

    return train_dataset, train_dataloader, dev_dataset, dev_dataloader

def train_epoch(train_dataloader, model, optimizer, loss_fn):
    for idx, (feature_batch, label_batch) in enumerate(train_dataloader):
        pred_batch = model(feature_batch)
        loss = loss_fn(pred_batch, label_batch)
        print(f'Batch {idx}, loss: {loss.data[0]}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    features_path = 'sudoku/data/2/features.pt'
    labels_path = 'sudoku/data/2/labels.pt'

    _, train_dataloader, _, dev_dataloader = build_dataloaders(features_path, labels_path)
    board_size = 4 # data/2 subfolder is 4x4 grids, data/3 subfolder is 9x9 grids

    # can replace the following with whatever other model you have (imported above)
    # all of them i think use the same loss function anyway
    model = OptNet(1, board_size, batch_sizeg**3, 40, q_penalty=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    n_epochs = 5
    for epoch_iter in range(n_epochs):
        print(f'Starting Epoch {epoch_iter+1}/{n_epochs}')
        train_epoch(train_dataloader, model, optimizer, loss_fn)


if __name__=='__main__':
  main();
