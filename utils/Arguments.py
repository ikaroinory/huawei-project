import argparse

import torch


class Arguments:
    def __init__(self):
        args = self.__parse_args()

        self.seed: int = args.seed

        self.model_path: str | None = args.model

        self.key_subsequence = args.key_subsequence
        self.mode = 'key_subsequence' if self.key_subsequence else 'normal'

        self.dtype = torch.float32 if args.dtype == 'float' else torch.float64
        self.device = args.device

        self.batch_size: int = args.batch_size
        self.epochs: int = args.epochs

        self.d_hidden: int = args.d_hidden
        self.d_ff: int = args.d_ff
        self.d_embedding: int = args.d_embedding

        self.num_heads: int = args.num_heads
        self.num_layers: int = args.num_layers

        self.test_size: int = args.test_size

        self.lr = args.lr

        self.early_stop: int = args.early_stop
        self.dropout: int = args.dropout

        self.log: bool = not args.nolog

    @staticmethod
    def __parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--seed', type=int, default=42)

        parser.add_argument('--model', type=str)

        parser.add_argument('-k', '--key_subsequence', action='store_true')

        parser.add_argument('--dtype', choices=['float', 'double'], default='float')
        parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')

        parser.add_argument('-b', '--batch_size', type=int, default=32)
        parser.add_argument('-e', '--epochs', type=int, default=20)

        parser.add_argument('--d_hidden', type=int, default=512)
        parser.add_argument('--d_ff', type=int, default=1024)
        parser.add_argument('--d_embedding', type=int, default=1024)

        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--num_layers', type=int, default=4)

        parser.add_argument('--test_size', type=float, default=0.2)

        parser.add_argument('-l', '--lr', type=float, default=0.0001)

        parser.add_argument('--early_stop', type=int, default=10)
        parser.add_argument('--dropout', type=float, default=0.1)

        parser.add_argument('--nolog', action='store_true')

        return parser.parse_args()
