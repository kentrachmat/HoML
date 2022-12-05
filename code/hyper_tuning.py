import json
import optuna
import torch
from custom_model import Net_2
from train_custom import train, evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def objective(trial: optuna.trial.Trial):
    EPOCHS = 20
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    ch_1 = trial.suggest_int("ch_1", low=8, high=16, step=2)
    ch_2 = trial.suggest_int("ch_2", low=ch_1, high=2*ch_1, step=4)
    ch_3 = trial.suggest_int("ch_3", low=ch_1, high=2*ch_2, step=4)
    fc_neur = trial.suggest_int("fc_neurons", low=int(1.5*ch_3), high=(2.5*ch_3), step=2)


    model = Net_2(fc_neurons=fc_neur, channels=[ch_1, ch_2, ch_3])
    model.to(device)

    for epoch in range(EPOCHS):
        print('Epoch:', epoch)
        train_loss = train(lr, model)
        valid_loss = evaluate(model)

        trial.report(valid_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return valid_loss


if __name__ == '__main__':
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    
    with open('../data/best_params.json', 'w') as outfile:
        outfile.write(json.dumps(best_params, indent=4))
