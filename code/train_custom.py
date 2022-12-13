import json
import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from custom_model import Net_2, FineTunedEffnet


PATH_TO_TRAIN = '../data/new_data/train'
PATH_TO_VALID = '../data/new_data/valid-lab'

train_set = datasets.ImageFolder(root=PATH_TO_TRAIN, transform=transforms.ToTensor())
valid_set = datasets.ImageFolder(root=PATH_TO_VALID, transform=transforms.ToTensor())

train_data = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
valid_data = DataLoader(valid_set, batch_size=16, shuffle=True, num_workers=2)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()


def train(lr, model):
    """Trains a given model with Adam optimizer
    and learning rate lr (only one epoch)

    Args:
        lr: learning rate
        model: model to train

    Returns:
        avg_loss: exponentially moving average
        over the batches of the training set
    """    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    avg_loss = None
    for batch_idx, (images, labels) in enumerate(train_data):

        images, labels =  images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if avg_loss is None:
            avg_loss = loss.data.item()
        else:
            avg_loss = 0.5*avg_loss + 0.5*loss.data.item()

        if batch_idx % 10 == 0:
            print(avg_loss)

    return avg_loss


def evaluate(model):
    """Evaluates a given model on the
    validation set

    Args:
        model (callable): the model to evaluate

    Returns:
        float: the crossentropy loss
    """    
    model.eval()
    avg_loss = 0
    accuracy = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(valid_data):
            images, labels =  images.to(device), labels.to(device)
            output = model(images)

            loss = criterion(output, labels)

            accuracy += (output.argmax(axis=1)==labels).sum()/len(valid_set)

            avg_loss += len(images)/len(valid_data)*loss.data.item()

    print('Accuracy:', accuracy)
    return avg_loss

def load_best_params():
    """
        Reads the parameters in data/best_params.json
        and returns a dictionary.
    """    
    with open('../data/best_params.json') as infile:
        best_params = json.load(infile)

    fc_neurons = best_params['fc_neurons']
    channels = [best_params[f'ch_{n}'] for n in [1,2,3]]
    lr = best_params['lr']

    params = {
        'lr': fc_neurons,
        'fc_neurons': fc_neurons,
        'channels': channels
    }
    return params

if __name__ == '__main__':
    # With these variables you can choose
    # what type of model to train:
    # - Manually inserting hyperparameters
    # - Using the best hyperparameters
    # - Finetuning EfficientNet

    RANDOM_PARAMS = False
    BEST_PARAMS = False
    EFF_NET = True

    if RANDOM_PARAMS:
        lr = 1e-3
        model_params = {
            'fc_neurons': 70,
            'channels': [16, 20, 28]
        }
        model = Net_2(**model_params)

    elif BEST_PARAMS:
        best_params = load_best_params()
        model = Net_2(best_params['fc_neurons'], 
                    best_params['channels'])
        lr = best_params['lr']

    elif EFF_NET:
        lr = 1e-4
        model = FineTunedEffnet()


    model.to(device)
    for epoch in range(5):
        train(lr, model)

    evaluate(model)