from torch.utils.data import Dataset
from os import listdir, path
from PIL import Image
import torch


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, verbose=False, num_iters=None, batch_size=1):
        """
        Args:
            root_dirs (string): A list of directories with all the images' folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dirs = root_dir
        self.transform = transform
        self.images_path = []
        self.images_path += [path.join(root_dir, file) for file in listdir(root_dir) if file.endswith(('png','jpg','jpeg','bmp'))]
        self.verbose = verbose
        self.num_iters = num_iters
        self.num_images = len(self.images_path)
        self.batch_size = batch_size

    def __len__(self):
        if self.num_iters is None:
            return len(self.images_path)
        return self.num_iters * self.batch_size

    def __getitem__(self, idx):
        if self.num_iters is None:
            img_name = self.images_path[idx]
        else:
            # idx = np.random.randint(0,self.num_images)
            idx = torch.randint(0,self.num_images,(1,)).numpy()[0]
            img_name = self.images_path[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.verbose:
            return image, img_name.split('/')[-1]

        return image