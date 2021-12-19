from PIL import Image
from torch.utils import data


class DataSet(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, args, is_train, data_path, data_label, transform):
        'Initialization'
        self.labels = data_label
        self.examples = data_path
        self.transform = transform
        self.image_dir = args['img_dir']
        self.args = args
        self.is_train = is_train


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.examples)

    def __getitem__(self, idx):
        'Generates one sample of data'
        id = self.examples[idx]
        # Convert to RGB to avoid png.
        X = Image.open(self.image_dir + id).convert('RGB')
        # X = Image.open(self.image_dir + id).convert('L')
        # X=rgb2gray(X)
        X = self.transform(X)
        label = self.labels[idx]

        return X, label
