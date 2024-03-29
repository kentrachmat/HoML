from tqdm import tqdm
import numpy as np
import torch
from test_dataset import TestDataset
from torch.utils.data import DataLoader
from custom_model import Net_2, FineTunedEffnet
from train_custom import load_best_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # automatically select on what device to run the code


def load_model_scratch():
    print('Loading model from scratch...')
    params = load_best_params()
    model = Net_2(params['fc_neurons'], 
                  params['channels'])
    model.load_state_dict(torch.load('../data/trained_model_scratch', map_location = 'cpu'))

    return model.to(device)


def load_model_effnet():
    print('Loading fine tuned effnet...')
    model = FineTunedEffnet()
    model.load_state_dict(torch.load('../data/trained_model_effnet'))

    return model.to(device)


def predict(data_loader, model):
    preds = []
    with torch.no_grad():
        for n, X in tqdm(enumerate(data_loader)): 
            X = X.to(device)
            X = X.float()
            y = model(X)
            preds.append(y.argmax(axis=1).cpu().numpy())
    return np.concatenate(preds, dtype=int)


if __name__ == '__main__':
    # Make predictions on the validation and test sets
    # to choose between the model just change the function
    # that gets called (either effnet or scratch)
    model = load_model_effnet()

    valid_data = TestDataset('../data/new_data/valid', 'valid')
    test_data = TestDataset('../data/new_data/test', 'test')

    valid_loader = DataLoader(valid_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)

    print('Predicting valid...')
    valid_predict = predict(valid_loader, model)
    print('Predicting test...')
    test_predict = predict(test_loader, model)

    np.savetxt('../results/Areal_valid.predict', valid_predict, fmt='%d')
    np.savetxt('../results/Areal_test.predict', test_predict, fmt='%d')
