import numpy as np
import torch
from test_dataset import TestDataset
from torch.utils.data import DataLoader
from custom_model import Net_2
from train_custom import load_best_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # automatically select on what device to run the code


def load_model():
    params = load_best_params()
    model = Net_2(params['fc_neurons'], 
                  params['channels'])
    model.load_state_dict(torch.load('../data/trained_model'))

    return model.to(device)


def predict(data_loader, model):
    preds = []
    with torch.no_grad():
        for n, X in enumerate(data_loader): 
            X = X.to(device)
            X = X.float()
            y = model(X)
            preds.append(y.argmax(axis=1).cpu().numpy())
    return np.concatenate(preds, dtype=int)


if __name__ == '__main__':

    model = load_model()

    valid_data = TestDataset('../data/new_data/valid', 'valid')
    test_data = TestDataset('../data/new_data/test', 'test')

    valid_loader = DataLoader(valid_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)

    valid_predict = predict(valid_loader, model)
    test_predict = predict(test_loader, model)

    np.savetxt('../results/Areal_valid.predict', valid_predict, fmt='%d')
    np.savetxt('../results/Areal_test.predict', test_predict, fmt='%d')





