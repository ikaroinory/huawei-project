import torch
from optuna import Trial


class OptunaArguments:
    def __init__(self, trial: Trial):
        self.seed: int = 42

        self.model_path: str | None = None

        self.key_subsequence: bool = True
        self.mode = 'key_subsequence' if self.key_subsequence else 'normal'

        self.dtype = torch.float32
        self.device = 'cuda'

        self.batch_size: int = 32
        self.epochs: int = 100

        self.d_hidden: int = trial.suggest_int('d_hidden', 32, 1024)
        self.d_ff: int = trial.suggest_int('d_ff', 32, 1024)
        self.d_embedding: int = trial.suggest_int('d_embedding', 32, 1024)

        self.num_heads: int = 8
        self.num_layers: int = trial.suggest_categorical('num_layers', [1, 2, 3, 4, 5, 6, 7, 8])

        self.test_size: float = 0.2

        self.lr: float = trial.suggest_float('lr', 1e-5, 1e-2, log=True)

        self.early_stop: int = 20
        self.dropout: float = trial.suggest_float('dropout', 0, 1)

        self.log: bool = True
