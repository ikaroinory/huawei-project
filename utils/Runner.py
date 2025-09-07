import copy
import json
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler, Subset
from tqdm import tqdm

from datasets import APIDataset
from models import Detector
from .Arguments import Arguments
from .Logger import Logger


class Runner:
    def __init__(self):
        self.args = Arguments()

        self.start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.__log_path = Path(f'logs/{self.args.mode}/{self.start_time}.log')
        self.__model_path = Path(f'saves/{self.args.mode}/{self.start_time}.pth')

        Logger.init(self.__log_path if self.args.log else None)

        Logger.info('Setting seed...')
        self.__set_seed()

        Logger.info('Loading data...')
        self.api_count = self.__get_api_count()
        self.api_version_list = [25, 26, 28]
        self.train_api_version = self.api_version_list[0]
        train_dataloader, test_dataloader = self.__get_dataloaders(self.train_api_version)

        self.__train_dataloader: DataLoader = train_dataloader
        self.__test_dataloader: DataLoader = test_dataloader

        Logger.info('Building model...')
        self.__model: Detector = Detector(
            d_input=self.api_count + 1,
            d_hidden=self.args.d_hidden,
            d_ff=self.args.d_ff,
            d_embedding=self.args.d_embedding,
            num_heads=self.args.num_heads,
            num_layers=self.args.num_layers,
            behavior_sequence_max_len=400,
            normal_sequence_max_len=4600,
            abnormal_sequence_max_len=700,
            dropout=self.args.dropout,
            dtype=self.args.dtype,
            device=self.args.device
        )
        self.__loss = BCEWithLogitsLoss()
        self.__optimizer = Adam(self.__model.parameters(), lr=self.args.lr)
        self.scaler = GradScaler()

    def __set_seed(self) -> None:
        random.seed(self.args.seed)

        np.random.seed(self.args.seed)

        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def __get_api_count() -> int:
        with open('data/processed/api_list.json', 'r') as f:
            api_list = json.load(f)
        return len(api_list)

    def __split_dataset(self, dataset: APIDataset) -> tuple[DataLoader, DataLoader]:
        dataset_size = int(len(dataset))
        train_dataset_size = int((1 - self.args.test_size) * dataset_size)
        test_dataset_size = int(self.args.test_size * dataset_size)

        test_start_index = random.randrange(train_dataset_size)

        indices = torch.arange(dataset_size)
        train_indices = torch.cat([indices[:test_start_index], indices[test_start_index + test_dataset_size:]])
        test_indices = indices[test_start_index:test_start_index + test_dataset_size]

        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)

        train_dataloader = DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=True, worker_init_fn=lambda _: self.__set_seed())
        test_dataloader = DataLoader(test_subset, batch_size=self.args.batch_size, shuffle=False, worker_init_fn=lambda _: self.__set_seed())

        return train_dataloader, test_dataloader

    def __get_dataloaders(self, api_version: int, only_test=False) -> tuple[DataLoader, DataLoader | None]:
        with open(f'data/processed/api{api_version}.pkl', 'rb') as file:
            df_train_data = pickle.load(file)
            df_train_data = pd.DataFrame(
                df_train_data,
                columns=['api_sequence', 'normal_key_api_sequence', 'abnormal_key_api_sequence', 'label']
            )

            x = np.array(df_train_data['api_sequence'].tolist())
            normal_key_api_sequence = np.array(df_train_data['normal_key_api_sequence'].tolist())
            abnormal_key_api_sequence = np.array(df_train_data['abnormal_key_api_sequence'].tolist())
            y = np.array(df_train_data['label'].tolist())

            dataset = APIDataset(x, normal_key_api_sequence, abnormal_key_api_sequence, y, dtype=self.args.dtype)

            if only_test:
                return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, sampler=SequentialSampler(dataset)), None
            else:
                return self.__split_dataset(dataset)

    @staticmethod
    def get_accuracy(pred_tensor: Tensor, label_tensor: Tensor):
        return accuracy_score(label_tensor.cpu(), pred_tensor.cpu())

    def __train_epoch(self) -> float:
        self.__model.train()

        total_loss = 0
        for behavior, normal, abnormal, label in tqdm(self.__train_dataloader):
            behavior = behavior.to(self.args.device)
            normal = normal.to(self.args.device)
            abnormal = abnormal.to(self.args.device)
            label = label.to(self.args.device)

            self.__optimizer.zero_grad()
            with autocast(self.args.device):
                output = self.__model(behavior, normal, abnormal).squeeze()

                loss = self.__loss(output, label)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.__optimizer)
            self.scaler.update()

            total_loss += loss.item() * behavior.shape[0]

        return total_loss / len(self.__train_dataloader.dataset)

    def __test_epoch(self, dataloader: DataLoader) -> tuple[float, Tensor, Tensor]:
        self.__model.eval()

        pred_list = []
        label_list = []

        total_loss = 0
        for behavior, normal, abnormal, label in tqdm(dataloader):
            behavior = behavior.to(self.args.device)
            normal = normal.to(self.args.device)
            abnormal = abnormal.to(self.args.device)
            label = label.to(self.args.device)

            with torch.no_grad():
                with autocast(self.args.device):
                    output = self.__model(behavior, normal, abnormal).squeeze()

                    loss = self.__loss(output, label)

                total_loss += loss.item() * behavior.shape[0]

                pred_labels = (torch.sigmoid(output) >= 0.5).long()

                pred_list.append(pred_labels)
                label_list.append(label)

        pred_tensor = torch.cat(pred_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0)

        return total_loss / len(dataloader.dataset), pred_tensor, label_tensor

    def __train(self) -> None:
        Logger.info('Training...')

        best_epoch = -1
        best_test_loss = float('inf')
        best_accuracy = 0
        best_model_weights = copy.deepcopy(self.__model.state_dict())
        patience_counter = 0

        for epoch in tqdm(range(self.args.epochs)):
            train_loss = self.__train_epoch()
            test_loss, pred_tensor, label_tensor = self.__test_epoch(self.__test_dataloader)

            accuracy = self.get_accuracy(pred_tensor, label_tensor)

            Logger.info(f'Epoch {epoch + 1}:')
            Logger.info(f' - Train loss: {train_loss:.8f}')
            Logger.info(f' - Test loss: {test_loss:.8f}')

            if test_loss < best_test_loss:
                best_epoch = epoch + 1

                best_test_loss = test_loss
                best_accuracy = accuracy

                best_model_weights = copy.deepcopy(self.__model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            Logger.info(f' - Current best epoch: {best_epoch}')

            if patience_counter >= self.args.early_stop:
                break

        self.__model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_weights, self.__model_path)

        Logger.info(f'Best epoch: {best_epoch}')
        Logger.info(f' - Test loss: {best_test_loss:.8f}')
        Logger.info(f' - Accuracy: {best_accuracy * 100:.2f}%')
        Logger.info(f'Model save to {self.__model_path}')

    def __evaluate(self, model_path: Path, api_version_list: list[int]) -> None:
        Logger.info('Evaluating...')

        self.__model.load_state_dict(torch.load(f'{model_path}', weights_only=True))

        Logger.info(f'Accuracy:')
        for api_version in api_version_list:
            if api_version == self.train_api_version:
                _, pred_tensor, label_tensor = self.__test_epoch(self.__train_dataloader)
                accuracy = self.get_accuracy(pred_tensor, label_tensor)
            else:
                dataloader, _ = self.__get_dataloaders(api_version, only_test=True)
                _, pred_tensor, label_tensor = self.__test_epoch(dataloader)
                accuracy = self.get_accuracy(pred_tensor, label_tensor)

            Logger.info(f' - API {api_version}: {accuracy * 100:.2f}%')

    def run(self) -> None:
        if self.args.model_path is None:
            self.__train()
            self.__evaluate(self.__model_path, self.api_version_list)
        else:
            self.__evaluate(Path(self.args.model_path), self.api_version_list)
