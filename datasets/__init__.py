from os.path import join
from datasets.cifar10 import CIFAR10
import torchvision.transforms as transforms

DATASETS = {
    'cifar10': CIFAR10,
}


def build_transform(train, img_size, crop_pad, flip):
    transform = []
    transform.append(transforms.Resize((img_size + crop_pad, img_size + crop_pad)))
    if train:
        transform.append(transforms.RandomCrop((img_size, img_size)))
        if flip:
            transform.append(transforms.RandomHorizontalFlip())
    else:
        transform.append(transforms.CenterCrop((img_size, img_size)))
    transform = transforms.Compose(transform)
    return transform


def build_data(data_name, data_path, train, transform, trigger):
    data = DATASETS[data_name](join(data_path, data_name), train, transform, trigger)
    return data
