import sys
import numpy as np
import torch
from dataset import get_clean_data_paths, get_DataLoader
from dataset import get_data_from_clean_data_paths
from train_utils import get_new_shuffle, Train, Model_evaluate
from model import Denoising_Model

train_batch_size = 64
dev_batch_size = 64
# part of train data for progress estimation
dev_part = 0.1
# starting learning rate
learning_rate = 0.001
layers = 15
# max number of epochs
max_epochs = 100


device = ''
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(f'Your device is "{device}"')


#example of path
# './val/val/'
#'/content/gdrive/MyDrive/Colab Notebooks/train/train/'
train_path = sys.argv[1]
#'/content/gdrive/MyDrive/Colab Notebooks/val/val/'
val_path = sys.argv[2]

tr_clean_paths_list = get_clean_data_paths(train_path)
val_clean_paths_list = get_clean_data_paths(val_path)

train_clean_data, train_noisy_data = get_data_from_clean_data_paths(tr_clean_paths_list[:])
val_clean_data, val_noisy_data = get_data_from_clean_data_paths(val_clean_paths_list[:])


train_clean_data, train_noisy_data = get_new_shuffle(train_clean_data, train_noisy_data)

dev_data_length = int(len(train_clean_data)*dev_part)
train_data_length = len(train_clean_data) - dev_data_length

train_dataloader = get_DataLoader(train_clean_data[:train_data_length],
                                  train_noisy_data[:train_data_length],
                                  train_batch_size)
dev_dataloader = get_DataLoader(train_clean_data[train_data_length:],
                                train_noisy_data[train_data_length:],
                                dev_batch_size)


model = Denoising_Model(layers)
model.to(device)

np.set_printoptions(suppress=True)
cur_lr = learning_rate * 0.1
best_model_epoch = 0
no_improve_epochs = 0
stop_flag_repeat_no_improve = 3
best_dev_loss = np.Inf

train_losses = []
dev_losses = []

for epoch in range(1, max_epochs + 1):
    train_loss = Train(model, train_dataloader, device, epoch, cur_lr)
    cur_dev_loss = Model_evaluate(model, dev_dataloader, device)
    train_losses.append(train_loss)
    dev_losses.append(cur_dev_loss)

    if cur_dev_loss < best_dev_loss:
        best_dev_loss = cur_dev_loss
        best_model_epoch = epoch
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        if no_improve_epochs == stop_flag_repeat_no_improve:
            print(f'Stop Training\n'
                  f"Best result of model -  'denoiser_epoch{best_model_epoch}.pth'")
            break

loaded_denoising_model = Denoising_Model(layers)
model_state_dict = torch.load(f'denoiser_epoch{best_model_epoch}.pth')
loaded_denoising_model.load_state_dict(model_state_dict)
loaded_denoising_model.to(device)

torch.save(model.state_dict(), 'denoiser_best.pth')

val_dataloader = get_DataLoader(val_clean_data, val_noisy_data, batch_size=1, shuffle=False)
val_loss = Model_evaluate(loaded_denoising_model, val_dataloader, device)
print(val_loss)

#best_model_epoch important value