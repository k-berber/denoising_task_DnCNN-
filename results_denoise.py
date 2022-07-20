import sys
import torch
from dataset import get_clean_data_paths, get_DataLoader
from dataset import get_data_from_clean_data_paths
from train_utils import Model_evaluate, Save_results
from model import Denoising_Model

test_path = sys.argv[1]
#'./val/val/'

device = ''
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(f'Your device is "{device}"')


test_clean_paths_list = get_clean_data_paths(test_path)
test_clean_data, test_noisy_data = get_data_from_clean_data_paths(test_clean_paths_list[:])
test_dataloader = get_DataLoader(test_clean_data, test_noisy_data, batch_size=1, shuffle=False)

loaded_denoising_model = Denoising_Model(15)


model_state_dict = torch.load('./denoiser_epoch4.pth', map_location=torch.device('cpu'))
#here you should indicate the best model


#print(model_state_dict)
loaded_denoising_model.load_state_dict(model_state_dict)
loaded_denoising_model.to(device);

test_loss = Model_evaluate(loaded_denoising_model, test_dataloader, device)
print(f'MSE = {test_loss:6.4}')

Save_results(loaded_denoising_model, test_clean_data, test_noisy_data, device)