import optuna
from optuna import Trial

from utils import Runner


def objective(trial: Trial) -> list[float]:
    runner = Runner(trial)
    acc_list = runner.run()
    return acc_list


if __name__ == '__main__':
    db_string = f'sqlite:///optuna.db'
    study = optuna.create_study()

    study.optimize(objective, n_trials=1000)
