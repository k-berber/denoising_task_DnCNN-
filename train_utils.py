import os
import numpy as np

import random
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Detect


def get_new_shuffle(list1, list2):
    """
    Shuffles 2 lists of the same length in coherent manner
    and returns them.
    """
    if len(list1) != len(list2):
        print('wrong len!')
        return -1
    new_order = list(range(len(list1)))
    random.shuffle(new_order)
    new_list1 = [list1[new_order[i]] for i in new_order]
    new_list2 = [list2[new_order[i]] for i in new_order]

    return new_list1, new_list2


def Train(model, dataloader, device, epoch, learning_rate):
    model.train()
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    summ_loss = 0.0
    processed_sample = 0
    processed_samples_report_period = 3500
    next_report = processed_samples_report_period

    print("Train dataset statistics: ")
    for batch_number, data in enumerate(dataloader):
        samples, targets = data[0], data[1]
        samples, targets = samples.to(device), targets.to(device)

        epoch_loss = 0
        epoch_acc = 0
        model.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        summ_loss += loss.item()

        # print statistics
        processed_sample += targets.shape[0]
        if processed_sample >= next_report:
            print(f'epoch = {epoch:4} '
                  f'mse_loss = {summ_loss / processed_sample:7.6f}')
            next_report += processed_samples_report_period

    torch.save(model.state_dict(), f'denoiser_epoch{epoch}.pth')
    return summ_loss / processed_sample


def Model_evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss(reduction='mean')
    summ_loss = 0.
    processed_sample = 0
    for batch_number, data in enumerate(dataloader):
        samples, targets = data[0], data[1]
        samples, targets = samples.to(device), targets.to(device)

        model_predictions = model(samples)
        targets = targets
        samples = samples
        mel_predictions = samples - model_predictions
        mel_targets = samples - targets

        summ_loss += criterion(mel_predictions, mel_targets).item()
        processed_sample += targets.shape[0]

    print(f"\nEval dataset statistics: "
          f"eval_mse_loss = {summ_loss / processed_sample:7.6f}\n")

    return summ_loss / processed_sample


def Save_results(model, clean_data, noisy_data, device):
    model.eval()
    dataset = Detect(clean_data, noisy_data)
    criterion = nn.MSELoss(reduction='mean')
    os.mkdir("model_predict")
    # cnt = 0
    for data_idx in range(len(dataset)):
        data = dataset[data_idx]
        sample, target = torch.tensor(data[0]).unsqueeze(0).float(), torch.tensor(data[1]).unsqueeze(0).float()
        sample, target = sample.to(device), target.to(device)

        prediction = model(sample)
        mel_prediction = sample - prediction
        mel_target = sample - target

        loss = criterion(mel_prediction, mel_target).item()
 
        mel_prediction = mel_prediction.squeeze(0).cpu().detach().numpy()  # <-- необходимо для сохранения

        file_name = f'model_predict/prediction_{data_idx}.npy'
        with open(file_name, 'wb') as f:
            np.save(f, mel_prediction)
        # cnt += 1
        pass
