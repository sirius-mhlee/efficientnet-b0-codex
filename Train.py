import copy
import numpy as np
import pandas as pd

import pickle

from tqdm.auto import tqdm

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import Config

if Config.fixed_randomness:
    import Util.Randomness as Randomness

from Model.Loss import CustomCrossEntropyLoss

from Util.CustomDataset import CustomDataset
from Util.Mapper import get_model_by_name
from Util.Metric import macro_f1_score

def main():
    # Data Load
    df = pd.read_csv('./Data/train.csv')
    if Config.test_run:
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(Config.test_run_data_size)

    le = preprocessing.LabelEncoder()
    df['artist'] = le.fit_transform(df['artist'].values)
    pickle.dump(le, open('./Output/encoder.pkl', 'wb'))

    df.sort_values(by=['id'], inplace=True)
    train_x = df['img_path'].values
    train_y = df['artist'].values

    # Define Device, Print Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define KFold, Transform
    if Config.use_fold:
        if Config.fixed_randomness:
            skf = StratifiedKFold(n_splits=Config.fold_k, shuffle=True, random_state=Config.seed)
        else:
            skf = StratifiedKFold(n_splits=Config.fold_k, shuffle=True)
    else:
        if Config.fixed_randomness:
            split_train_idx, split_val_idx = train_test_split(np.arange(len(train_x)), test_size=0.2, shuffle=True, random_state=Config.seed)
        else:
            split_train_idx, split_val_idx = train_test_split(np.arange(len(train_x)), test_size=0.2, shuffle=True)

    train_transform = v2.Compose([
                                v2.Resize((Config.img_size, Config.img_size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
                                v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ])

    val_transform = v2.Compose([
                                v2.Resize((Config.img_size, Config.img_size), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
                                v2.ToImage(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ])

    for idx, train_model_name in enumerate(Config.train_model_name_list, start=1):
        print()
        print(f'Model: {idx}, Name: {train_model_name}')
        if Config.print_model:
            print()
            print(get_model_by_name(train_model_name))
        print()

        fold_best_epoch = []
        fold_best_loss = []
        fold_best_score = []

        if Config.use_fold:
            data_generator = skf.split(train_x, train_y)
        else:
            data_generator = [(split_train_idx, split_val_idx)]

        for fold, (train_idx, val_idx) in enumerate(data_generator, start=1):
            print(f'Fold: {fold}')

            # Define Dataset, Dataloader
            fold_train_x = train_x[train_idx]
            fold_train_y = train_y[train_idx]

            fold_val_x = train_x[val_idx]
            fold_val_y = train_y[val_idx]

            train_dataset = CustomDataset(fold_train_x, fold_train_y, train_transform)
            val_dataset = CustomDataset(fold_val_x, fold_val_y, val_transform)

            if Config.fixed_randomness:
                train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False, worker_init_fn=Randomness.worker_init_fn, generator=Randomness.generator)
                val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False, worker_init_fn=Randomness.worker_init_fn, generator=Randomness.generator)
            else:
                train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False)
                val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=Config.data_loader_worker_num, pin_memory=True, drop_last=False)

            # Define Model, Criterion, Optimizer, Scheduler
            model = get_model_by_name(train_model_name)
            model = nn.DataParallel(model)
            model.to(device)

            criterion = CustomCrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

            # Train
            train_loss_list = []
            val_loss_list = []

            best_epoch = 0
            best_loss = np.inf
            best_score = 0.0
            best_model = copy.deepcopy(getattr(model, 'module', model).state_dict())

            rng = np.random.default_rng()

            print()
            for epoch in range(1, Config.epoch + 1):
                model.train()
                train_loss_sum = 0.0
                train_seen = 0

                for inputs, targets in tqdm(train_loader):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True).long()

                    optimizer.zero_grad(set_to_none=True)

                    if Config.use_mixup:
                        if Config.fixed_randomness:
                            lambda_value = Randomness.rng.beta(1.0, 1.0)
                        else:
                            lambda_value = rng.beta(1.0, 1.0)

                        mixed_index = torch.randperm(inputs.size(0)).to(device, non_blocking=True)

                        mixed_inputs = lambda_value * inputs + (1 - lambda_value) * inputs[mixed_index]
                        target_a, target_b = targets, targets[mixed_index]

                        outputs = model(mixed_inputs)
                        loss = lambda_value * criterion(outputs, target_a) + (1 - lambda_value) * criterion(outputs, target_b)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()

                    batch_size = inputs.size(0)
                    train_loss_sum += loss.detach().item() * batch_size
                    train_seen += batch_size

                model.eval()
                val_loss_sum = 0.0
                val_seen = 0

                val_pred_list = []
                val_target_list = []

                with torch.no_grad():
                    for inputs, targets in tqdm(val_loader):
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True).long()

                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        batch_size = inputs.size(0)
                        val_loss_sum += loss.detach().item() * batch_size
                        val_seen += batch_size

                        _, preds = torch.max(outputs, 1)

                        val_pred_list.extend(preds.detach().cpu().tolist())
                        val_target_list.extend(targets.detach().cpu().tolist())

                epoch_train_loss = train_loss_sum / max(1, train_seen)
                epoch_val_loss = val_loss_sum / max(1, val_seen)

                train_loss_list.append(epoch_train_loss)
                val_loss_list.append(epoch_val_loss)

                val_score = macro_f1_score(val_target_list, val_pred_list)

                if scheduler is not None:
                    if hasattr(scheduler, 'step') and 'ReduceLROnPlateau' in type(scheduler).__name__:
                        scheduler.step(epoch_val_loss)
                    else:
                        scheduler.step()
                    epoch_lr = optimizer.param_groups[0]['lr']
                else:
                    epoch_lr = Config.learning_rate

                print(f'Epoch: {epoch}, Learning Rate: {epoch_lr:.6f}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Score: {val_score:.4f}')
                print()

                if epoch_val_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_val_loss
                    best_score = val_score
                    best_model = copy.deepcopy(getattr(model, 'module', model).state_dict())

            fold_best_epoch.append(best_epoch)
            fold_best_loss.append(best_loss)
            fold_best_score.append(best_score)

            torch.save({'epoch': best_epoch,
                        'loss': best_loss,
                        'score': best_score,
                        'name': train_model_name,
                        'model_state_dict': best_model},
                        f'./Output/{train_model_name}_fold_{fold}_result.pt')

        print(f'Fold Best Epoch: {fold_best_epoch}')
        print(f'Fold Best Loss: {fold_best_loss}')
        print(f'Fold Best Score: {fold_best_score}')

    print()

if __name__ == '__main__':
    main()
