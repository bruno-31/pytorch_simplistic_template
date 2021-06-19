from torch.utils.data import Dataset
from os import listdir, path
from PIL import Image


class DenoisingDataset(Dataset):
    def __init__(self, root_dir, transform=None, verbose=False):
        """
        Args:
            root_dirs (string): directory with all the images' folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images_path = []
        self.images_path += [path.join(root_dir, file) for file in listdir(root_dir) if file.endswith(('png','jpg','jpeg','bmp'))]
        self.verbose = verbose

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_name = self.images_path[idx]
        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.verbose:
            return image, img_name.split('/')[-1]

        return image
