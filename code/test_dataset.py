from torch.utils.data import Dataset
from pathlib import Path
from torchvision.io import read_image

class TestDataset(Dataset):

    def __init__(self, path, name):
        self.path = path
        self.name = name
    
    def __len__(self):
        return len(list(Path(self.path).iterdir()))

    def __getitem__(self, idx):
        image_path = self.path+f'/{self.name}_{idx}.png'
        img = read_image(image_path)/255.
        return img

        
if __name__ == '__main__':

    dataset = TestDataset('../data/new_data/test', 'test')

    print(len(dataset))

    print(dataset[0])