import pandas as pd

import pickle

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2

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
    test_transform = v2.Compose([
                                v2.Resize((Config.img_size, Config.img_size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
                                v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ])

    test_dataset = CustomDataset(test_x, None, test_transform)

    if Config.fixed_randomness:
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False, worker_init_fn=Randomness.worker_init_fn, generator=Randomness.generator)
    else:
        test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False)

    # Define Modellist, Print Modellist
    model_list = []
    for idx, test_input_model in enumerate(Config.test_input_model_list, start=1):
        ckpt = torch.load(f'./Output/{test_input_model}')

        test_model_name = ckpt['name']

        print()
        print(f'Model: {idx}, Name: {test_model_name}')

        model = get_model_by_name(test_model_name)
        if Config.print_model:
            print()
            print(model)
        print()
        print('Epoch: {}, Val Loss: {:.4f}, Val Score: {:.4f}'.format(ckpt['epoch'], ckpt['loss'], ckpt['score']))

        model.load_state_dict(ckpt['model_state_dict'])
        model = nn.DataParallel(model)
        model.to(device)
        model.eval()
        model_list.append(model)

    # Test
    test_pred_list = []

    print()
    with torch.inference_mode():
        for inputs in tqdm(test_loader):
            inputs = inputs.to(device, non_blocking=True)

            logits_sum = torch.zeros((inputs.size(0), Config.class_num)).to(device)
            for model in model_list:
                outputs = model(inputs)
                logits_sum = torch.add(logits_sum, outputs)

            logits_avg = torch.div(logits_sum, len(model_list))
            probs = F.softmax(logits_avg, dim=1)
            _, preds = torch.max(probs, 1)

            test_pred_list.extend(preds.detach().cpu().tolist())

    print()

    submit = pd.read_csv('./Data/sample_submission.csv')
    if Config.test_run:
        submit = submit.head(Config.test_run_data_size)

    submit['artist'] = le.inverse_transform(test_pred_list)
    submit.to_csv('./Output/submit.csv', index=False)

if __name__ == '__main__':
    main()
