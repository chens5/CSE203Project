import torch
import time
import numpy as np
from models.optnet_cvxpylayers import OptNet, LPLayer

torch.manual_seed(309)
np.random.seed(484)
CUDA_DEVICE='cuda:1'

def build_dataloaders(features_path, labels_path, train_batch_size=4, dev_percent=0.1):
    with open(features_path, 'rb') as fp:
        features = torch.load(fp).double()
    with open(labels_path, 'rb') as fp:
        labels = torch.load(fp).double()

    n_instances, n_features = features.size(0), int(np.prod(features.size()[1:]))
    n_train = int(n_instances*(1.-dev_percent))
    n_test = n_instances-n_train

    # dataset is small so move all to cuda up front
    if torch.cuda.is_available():
        train_tensors = [features[:n_train].to(CUDA_DEVICE), labels[:n_train].to(CUDA_DEVICE)]
        dev_tensors = [features[n_train:].to(CUDA_DEVICE), labels[n_train:].to(CUDA_DEVICE)]
    else:
        train_tensors = [features[:n_train], labels[:n_train]]
        dev_tensors = [features[n_train:], labels[n_train:]]

    train_dataset = torch.utils.data.TensorDataset(*train_tensors)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)

    dev_dataset = torch.utils.data.TensorDataset(*dev_tensors)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=0)

    return train_dataset, train_dataloader, dev_dataset, dev_dataloader

def train_epoch(train_dataloader, model, optimizer, loss_fn, is_cuda):
    for idx, (feature_batch, label_batch) in enumerate(train_dataloader):
        start_time = time.time()
        # only needed if full dataset not loaded to gpu at once
        # if is_cuda:
        #     feature_batch = feature_batch.to('cuda')
        #     label_batch = label_batch.to('cuda')

        pred_batch = model(feature_batch)
        loss = loss_fn(pred_batch, label_batch)
        end_time = time.time()
        print(f'Batch {idx}/{len(train_dataloader)}, loss: {loss.item()}, batch_time_seconds: {end_time - start_time}', flush=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def dev_eval(dev_dataloader, model, loss_fn, is_cuda):
    dev_loss_total = 0.0
    dev_instances = 0
    start_time = time.time()
    nErr = 0
    with torch.no_grad():
        for idx, (feature_batch, label_batch) in enumerate(dev_dataloader):
            # only needed if full dataset not loaded to gpu at once
            # if is_cuda:
            #     feature_batch = feature_batch.to('cuda')
            #     label_batch = label_batch.to('cuda')

            pred_batch = model(feature_batch)
            loss = loss_fn(pred_batch, label_batch)
            dev_loss_total += loss.item()
            dev_instances += feature_batch.size(0)
            nErr += computeErr(pred_batch.cpu().data)

    end_time = time.time()

    test_err = nErr/dev_instances
    print(f'Avg. dev loss over {dev_instances} instances: {dev_loss_total / dev_instances}, Avg. error: {test_err}, no_grad_elapsed_seconds: {end_time - start_time}', flush=True)

def computeErr(pred):
    batchSz = pred.size(0)
    nsq = int(pred.size(1))
    n = int(np.sqrt(nsq))
    s = (nsq-1)*nsq//2 # 0 + 1 + ... + n^2-1
    I = torch.max(pred, 3)[1].squeeze().view(batchSz, nsq, nsq)

    def invalidGroups(x):
        valid = (x.min(1)[0] == 0)
        valid *= (x.max(1)[0] == nsq-1)
        valid *= (x.sum(1) == s)
        return ~valid

    boardCorrect = torch.ones(batchSz).type_as(pred)
    for j in range(nsq):
        # Check the jth row and column.
        boardCorrect[invalidGroups(I[:,j,:])] = 0
        boardCorrect[invalidGroups(I[:,:,j])] = 0

        # Check the jth block.
        row, col = n*(j // n), n*(j % n)
        M = invalidGroups(I[:,row:row+n,col:col+n].contiguous().view(batchSz,-1))
        boardCorrect[M] = 0

        if boardCorrect.sum() == 0:
            return batchSz

    return batchSz-boardCorrect.sum().item()

def main():
    board_size = 4 # data/2 subfolder is 4x4 grids, data/3 subfolder is 9x9 grids
    train_batch_size = 150
    n_epochs = 10
    learning_rate = 1e-2

    if board_size == 9:
        features_path = 'sudoku/data/3/features.pt'
        labels_path = 'sudoku/data/3/labels.pt'
    elif board_size == 4:
        features_path = 'sudoku/data/2/features.pt'
        labels_path = 'sudoku/data/2/labels.pt'
    else:
        raise Exception('Invalid board_size, must be 4 or 9')

    _, train_dataloader, _, dev_dataloader = build_dataloaders(features_path, labels_path, train_batch_size=train_batch_size)

    # can replace the following with whatever other model you have (imported above)
    # all of them i think use the same loss function anyway
    #model = OptNet(1, board_size, g_dim=board_size**3-board_size, a_dim=40, q_penalty=0.1)
    model = LPLayer(1, board_size, a_dim=40, q_penalty=0.1)

    is_cuda = False
    if torch.cuda.is_available():
        is_cuda = True
        model.to(CUDA_DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for epoch_iter in range(n_epochs):
        print(f'Starting Epoch {epoch_iter+1}/{n_epochs}', flush=True)
        start_time = time.time()
        train_epoch(train_dataloader, model, optimizer, loss_fn, is_cuda)
        end_time = time.time()
        print(f'Finished Epoch {epoch_iter+1} in {end_time-start_time} seconds', flush=True)

        dev_eval(dev_dataloader, model, loss_fn, is_cuda)

if __name__=='__main__':
  main();
