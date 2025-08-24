import pandas as pd

import pickle

from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

from Util.CustomDataset import CustomDataset
from Util.Mapper import get_model_by_name

def main():
    # Data Load
    df = pd.read_csv('./Data/test.csv')
    if Config.test_run:
        #df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(Config.test_run_data_size)

    le = pickle.load(open('./Output/encoder.pkl', 'rb'))

    test_x = df['img_path'].values

    # Define Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define Transform, Dataset, Dataloader
    test_transform = A.Compose([
                                A.Resize(Config.img_size, Config.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_dataset = CustomDataset(test_x, None, test_transform)

    if Config.fixed_randomness:
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False, worker_init_fn=Randomness.worker_init_fn, generator=Randomness.generator)
    else:
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False)

    # Define Modellist, Print Modellist
    model_list = []
    for idx, test_input_model in enumerate(Config.test_input_model_list):
        idx += 1
        
        ckpt = torch.load('./Output/{}'.format(test_input_model))

        test_model_name = ckpt['name']

        print()
        print('Model: {}, Name: {}'.format(idx, test_model_name))

        model = get_model_by_name(test_model_name)
        if Config.print_model:
            print()
            print(model)
        print()
        print('Epoch: {}, Val Loss: {:.4}, Val Score: {:.4}'.format(ckpt['epoch'], ckpt['loss'], ckpt['score']))

        model.load_state_dict(ckpt['model_state_dict'])
        model = nn.DataParallel(model)
        model.to(device)
        model.eval()
        model_list.append(model)

    # Test
    test_pred_list = []

    print()
    with torch.no_grad():
        for input in tqdm(iter(test_loader)):
            input = input.to(device)

            percent = torch.zeros((input.size(0), Config.class_num)).to(device)
            for model in model_list:
                output = model(input)
                percent = torch.add(percent, F.softmax(output, dim=1))

            percent = torch.div(percent, len(model_list))
            _, pred = torch.max(percent, 1)

            test_pred_list.extend(pred.detach().cpu().numpy().tolist())

    print()

    submit = pd.read_csv('./Data/sample_submission.csv')
    if Config.test_run:
        submit = submit.head(Config.test_run_data_size)

    submit['artist'] = le.inverse_transform(test_pred_list)
    submit.to_csv('./Output/submit.csv', index=False)

if __name__ == '__main__':
    main()
