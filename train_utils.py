import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Detect


def crop_function(arr1, arr2, step):
    start_cut_range = arr1.shape[1] - step
    start_rand = np.random.randint(0, start_cut_range + 1)

    return (arr1[:, start_rand: start_rand + step], arr2[:, start_rand: start_rand + step])


def match_function(batch):
    min_len = 100000

    # extract data from input batch
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    for elem in data:
        min_len = min(min_len, elem.shape[1])

    temp_pairs = [crop_function(elem[0], elem[1], min_len) for elem in zip(data, targets)]
    data = [elem[0] for elem in temp_pairs]
    targets = [elem[1] for elem in temp_pairs]
    return [torch.tensor(data).float(), torch.tensor(targets).float()]


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


def Visualize(model, clean_data, noisy_data, device, list_of_indeces_to_check):
    model.eval()
    dataset = Detect(clean_data, noisy_data)
    criterion = nn.MSELoss(reduction='mean')
    # print('len dataset', len(dataset))
    for data_idx in list_of_indeces_to_check:
        data = dataset[data_idx]
        sample, target = torch.tensor(data[0]).unsqueeze(0).float(), torch.tensor(data[1]).unsqueeze(0).float()
        sample, target = sample.to(device), target.to(device)

        prediction = model(sample)
        mel_prediction = sample - prediction
        mel_target = sample - target

        loss = criterion(mel_prediction, mel_target).item()

        sample = sample.squeeze(0).cpu().detach().numpy()
        mel_prediction = mel_prediction.squeeze(0).cpu().detach().numpy()
        mel_target = mel_target.squeeze(0).cpu().detach().numpy()
        print(f'spectrogram {data_idx}, loss={loss:5.1}\n')

        plt.figure(figsize=(15, 15))
        plt.subplot(311)
        plt.title(f'Noisy image {data_idx}')
        sns.heatmap(sample, cmap="Blues")
        plt.subplot(312)
        plt.title(f'Predicted image {data_idx}')
        sns.heatmap(mel_prediction, cmap="Blues")
        plt.subplot(313)
        plt.title(f'Clean image {data_idx}')
        sns.heatmap(mel_target, cmap="Blues")
        pass
