import time
import os
import numpy as np
import torch
import torch.nn as nn
from data.data import FSD50K_train_dataset, FSD50K_test_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from eval_metrics import calculate_stats, calculate_overall_lwlrap


def train_(device, model, optimizer, criterion, train_dataload, graph):
    """Train routine"""
    model.train()
    total_train_loss = 0
    if graph is not None:
        graph = graph.to(device)
    for x_data, y_data in tqdm(train_dataload):
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        x_data = x_data.unsqueeze(1)  # add channel dimension
        optimizer.zero_grad()
        train_outputs = model(x_data, graph)
        train_loss = criterion(train_outputs, y_data)
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss.item()
    total_train_loss = (total_train_loss / len(train_dataload))

    return total_train_loss


def get_train_data(par):
    train_ds = FSD50K_train_dataset(par)
    train_dl = DataLoader(train_ds,
                          num_workers=par['num_workers'],
                          shuffle=True,
                          batch_size=par['batch_size'],
                          pin_memory=False,
                          drop_last=True)
    return train_dl


def get_test_data(par):
    return FSD50K_test_dataset(par)


def start_train(model, train_dataload, graph, par):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loss_list = []
    ckptfile = os.path.join(par['logs_path'], 'model.pth')
    e_tot = par['epochs']
    best_loss = 1e10

    if par['pre_train_path'] != '':
        model.load_state_dict(torch.load(par['pre_train_path']))
        print('Pretrained model loaded')

    model = model.to(device)

    if par['class_weights'] is not None:
        print('Use weighted BCEWithLogitsLoss')
        cw = torch.from_numpy(np.load(par['class_weights'])).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=cw)
    else:
        print('Use unweighted BCEWithLogitsLoss')
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=par['learning_rate'], weight_decay=5e-4)
    # TODO: parameterize schedulers
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5, verbose=par['verbose'])
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40], gamma=0.5, verbose=par['verbose'])  # multi-step lr reduction

    print(f'Start training with {device}...')

    for e in range(e_tot):
        t1 = time.time()

        epoch_train_loss = train_(device, model, optimizer, criterion, train_dataload, graph)
        if par['verbose']:
            print('Epoch [{}/{}] - Train Loss:{:.4f} - Elapsed time: {:.4f} s'.format(e + 1, e_tot, epoch_train_loss, time.time() - t1))

        # reduce LR according to scheduler policy
        # lr_scheduler.step()

        # track the train loss and accuracy
        train_loss_list.append(epoch_train_loss)

        # save net model only if loss decreases
        if epoch_train_loss < best_loss:
            torch.save(model.state_dict(), ckptfile)
            print(f'model checkpoint saved, previous loss: {best_loss}, current loss: {epoch_train_loss}')
            best_loss = epoch_train_loss

        # save performance values to csv
        perf_dict = {'TRAIN_LOSS': train_loss_list}
        df = pd.DataFrame(perf_dict)
        df.to_csv(os.path.join(par['logs_path'], 'loss.csv'))

    print('-' * 20)
    print('done')


def start_test(model, test_dataset, graph, par):
    """
    Clip-wise test
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load pretrained model
    model.load_state_dict(torch.load(os.path.join(par['logs_path'], 'model.pth')))
    model = model.to(device).eval()
    print('Pretrained model loaded')
    if graph is not None:
        graph = graph.to(device)
    print(f'Start test with {device}...')
    test_predictions = []
    test_gts = []  # ground truth
    for idx in tqdm(range(len(test_dataset))):
        with torch.no_grad():
            x_data, y_data = test_dataset[idx]
            x_data = x_data.to(device)
            y_pred = model(x_data, graph)  # forward pass
            y_pred = y_pred.mean(0).unsqueeze(0)
        test_predictions.append(y_pred.detach().cpu().numpy()[0])
        test_gts.append(y_data.detach().cpu().numpy()[0])
    test_predictions = np.asarray(test_predictions).astype('float32')
    test_gts = np.asarray(test_gts).astype('int32')

    APs, AUCs, d_prime = calculate_stats(test_gts, test_predictions)
    lwl_rap = calculate_overall_lwlrap(test_gts, test_predictions)
    np.save(os.path.join(par['logs_path'], 'APs.npy'), APs)
    np.save(os.path.join(par['logs_path'], 'AUCs.npy'), AUCs)
    m_APs = np.mean(APs)
    m_AUCs = np.mean(AUCs)
    print(f'mAP: {m_APs:.6f}')
    print(f'mAUC: {m_AUCs:.6f}')
    print(f'dprime: {d_prime:.6f}')
    print(f'lwl_rap: {lwl_rap:.6f}')

    # save metric values to csv
    test_dict = {'mAP': m_APs,
                 'mAUC': m_AUCs,
                 'dPrime': d_prime,
                 'lwl_rap': lwl_rap
                 }
    df = pd.DataFrame(data=list(test_dict.values()), index=list(test_dict.keys()), columns=None)
    df.to_csv(os.path.join(par['logs_path'], 'test_result.csv'))
