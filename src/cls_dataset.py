import random
from PIL import Image
from datasets import load_dataset

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import Compose, Resize
import torchvision.transforms.functional as F


class TinyImageNetFiltered(Dataset):
    def __init__(self, dataset, selected_class, forward_transforms):
        # Filter the dataset with the selected class
        if selected_class is not None:
            self.image_list = [item['image'] for item in dataset if
                               item['label'] == selected_class]
        else:
            self.image_list = [item['image'] for item in dataset]
        self.forward_transforms = forward_transforms
        self.tot_images = len(self.image_list)
        print(f'Tiny imagenet initialized with {self.tot_images} images')

    def __getitem__(self, index):
        im = self.image_list[index].convert('RGB')
        im = self.forward_transforms(im)
        return im

    def __len__(self):
        return self.tot_images


class RandomRotate90:
    def __call__(self, img):
        k = random.randint(0, 3)
        return F.rotate(img, angle=90 * k)


def get_tiny_imagenet_data_loader(image_size, batch_size, selected_class):
    # Get Tiny Imagenet
    dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    # Define forward transforms
    forward_transforms = Compose([
        Resize((image_size, image_size)),
        RandomRotate90(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    # Only select images with label zero
    tiny_imagenet = TinyImageNetFiltered(dataset,
                                         selected_class,
                                         forward_transforms)

    # Dataloader
    data_loader = DataLoader(tiny_imagenet, batch_size=batch_size, shuffle=True)
    return data_loader


if __name__ == '__main__':
    image_size = 64
    batch_size = 32
    selected_class = 0
    tiny_imagenet_loader = get_tiny_imagenet_data_loader(image_size, batch_size, selected_class)
