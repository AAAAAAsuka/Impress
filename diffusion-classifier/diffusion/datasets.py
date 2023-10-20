import os.path as osp
from torchvision import datasets
from diffusion.utils import DATASET_ROOT, get_classes_templates
from diffusion.dataset.objectnet import ObjectNetBase
from diffusion.dataset.imagenet import ImageNet as ImageNetBase
from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import re
import jsonlines
class MNIST(datasets.MNIST):
    """Simple subclass to override the property"""
    class_to_idx = {str(i): i for i in range(10)}

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

class diffBreakDataset_all_artist(Dataset):
    def __init__(self, root_dir, subfolders, all_style_data, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.style_data = read_jsonl(root_dir + "/style.jsonl")

        self.class_to_idx = {}
        self.file_to_class = {}

        # fill class_to_idx
        for i in range(len(all_style_data)):
            self.class_to_idx[all_style_data[i]] = i

        for subfolder in subfolders:
            artist = subfolder.split("/")[0]
            for style in self.style_data:
                if style['artist'] == artist:
                    true_style = re.sub('_', ' ', style['style'])
                    break

            folder_path = os.path.join(root_dir, subfolder)
            file_paths = glob.glob(folder_path + '/*.png')
            # fill file_to_class
            for file_path in file_paths:
                self.file_to_class[file_path] = true_style

            self.image_paths.extend(file_paths)
            self.labels.extend([self.class_to_idx[true_style]] * len(file_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

class diffBreakDataset_single(Dataset):
    def __init__(self, root_dir, subfolders, all_style_data, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.style_data = read_jsonl(root_dir + "/style.jsonl")

        self.class_to_idx = {}
        self.file_to_class = {}

        # fill class_to_idx
        for i in range(len(all_style_data)):
            self.class_to_idx[all_style_data[i]] = i

        for subfolder in subfolders:
            artist = subfolder.split("/")[0]
            for style in self.style_data:
                if style['artist'] == artist:
                    true_style = re.sub('_', ' ', style['style'])
                    break

            folder_path = os.path.join(root_dir, subfolder)
            file_paths = glob.glob(folder_path + '/*.png')
            # fill file_to_class
            for file_path in file_paths:
                self.file_to_class[file_path] = true_style

            self.image_paths.extend(file_paths)
            self.labels.extend([self.class_to_idx[true_style]] * len(file_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_diff_break_target_dataset(args, transform=None):
    if args.test_data == "clean":
        data_dir = f"{args.artist}/clean/test/{args.ft_step}/"
    elif args.test_data == "adv":
        data_dir = f"{args.artist}/{args.adv_para}/test/{args.ft_step}_transNum{args.trans_num}_seed{args.manual_seed}/"
    elif args.test_data == "pur":
        data_dir = f"{args.artist}/{args.pur_para}/test/{args.ft_step}_transNum{args.trans_num}_seed{args.manual_seed}/"
    else:
        data_dir = args.test_data


    image_class = ['Color_Field_Painting', 'Contemporary_Realism', 'Naive_Art_Primitivism', 'Post_Impressionism',
                   'Art_Nouveau_Modern', 'Analytical_Cubism', 'Impressionism', 'Expressionism', 'Action_painting',
                   'Early_Renaissance', 'Cubism', 'Mannerism_Late_Renaissance', 'Fauvism', 'Minimalism',
                   'New_Realism',
                   'Northern_Renaissance', 'Synthetic_Cubism', 'Rococo', 'Ukiyo_e', 'High_Renaissance', 'Symbolism',
                   'Pop_Art', 'Pointillism', 'Baroque', 'Romanticism', 'Realism', 'Abstract_Expressionism', 'Anime',
                   'Caricature', 'Cartoon', 'Picture Books', 'Comics', 'Commercial Art', 'Concept Art', 'Fantasy',
                   'Fashion', 'Fine art', 'Line art', 'Retro']
    for index, style in enumerate(image_class):
        image_class[index] = re.sub('_', ' ', style)
    subfolders = [data_dir]
    dataset = diffBreakDataset_single(root_dir="/home/Asuka/wikiart/preprocessed_data/", subfolders=subfolders,
                                      all_style_data=image_class, transform=transform)
    assert hasattr(dataset, "class_to_idx"), f"Dataset {args.test_data} does not have a class_to_idx attribute."
    assert hasattr(dataset, "file_to_class"), f"Dataset {args.test_data} does not have a file_to_class attribute."
    return dataset, data_dir

def get_target_dataset(name: str, train=False, transform=None, target_transform=None):
    """Get the torchvision dataset that we want to use.
    If the dataset doesn't have a class_to_idx attribute, we add it.
    Also add a file-to-class map for evaluation
    """

    if name == "cifar10":
        dataset = datasets.CIFAR10(root=DATASET_ROOT, train=train, transform=transform,
                                   target_transform=target_transform, download=True)
    elif name == "stl10":
        dataset = datasets.STL10(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                 target_transform=target_transform, download=True)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
    elif name == "pets":
        dataset = datasets.OxfordIIITPet(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                         target_transform=target_transform, download=True)

        # lower case every key in the class_to_idx
        dataset.class_to_idx = {k.lower(): v for k, v in dataset.class_to_idx.items()}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._images, dataset._labels)}
    elif name == "flowers":
        dataset = datasets.Flowers102(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                      target_transform=target_transform, download=True)
        classes = list(get_classes_templates('flowers')[0].keys())  # in correct order
        dataset.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._image_files, dataset._labels)}
    elif name == "aircraft":
        dataset = datasets.FGVCAircraft(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                        target_transform=target_transform, download=True)

        # replace backslash with underscore -> need to be dirs
        dataset.class_to_idx = {
            k.replace('/', '_'): v
            for k, v in dataset.class_to_idx.items()
        }

        dataset.file_to_class = {
            fn.split("/")[-1].split(".")[0]: lab
            for fn, lab in zip(dataset._image_files, dataset._labels)
        }
        # dataset.file_to_class = {
        #     fn.split("/")[-1].split(".")[0]: lab
        #     for fn, lab in zip(dataset._image_files, dataset._labels)
        # }

    elif name == "food":
        dataset = datasets.Food101(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                   target_transform=target_transform, download=True)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }
    elif name == "eurosat":
        if train:
            raise ValueError("EuroSAT does not have a train split.")
        dataset = datasets.EuroSAT(root=DATASET_ROOT, transform=transform, target_transform=target_transform,
                                   download=True)
    elif name == 'imagenet':
        assert not train
        base = ImageNetBase(transform, location=DATASET_ROOT)
        dataset = datasets.ImageFolder(root=osp.join(DATASET_ROOT, 'imagenet/val'), transform=transform,
                                       target_transform=target_transform)
        dataset.class_to_idx = None  # {cls: i for i, cls in enumerate(base.classnames)}
        dataset.classes = base.classnames
        dataset.file_to_class = None
    elif name == 'objectnet':
        base = ObjectNetBase(transform, DATASET_ROOT)
        dataset = base.get_test_dataset()
        dataset.class_to_idx = dataset.label_map
        dataset.file_to_class = None  # todo
    elif name == "caltech101":
        if train:
            raise ValueError("Caltech101 does not have a train split.")
        dataset = datasets.Caltech101(root=DATASET_ROOT, target_type="category", transform=transform,
                                      target_transform=target_transform, download=True)

        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.categories)}
        dataset.file_to_class = {str(idx): dataset.y[idx] for idx in range(len(dataset))}
    elif name == "mnist":
        dataset = MNIST(root=DATASET_ROOT, train=train, transform=transform, target_transform=target_transform,
                        download=True)

    else:
        raise ValueError(f"Dataset {name} not supported.")

    if name in {'mnist', 'cifar10', 'stl10', 'aircraft'}:
        dataset.file_to_class = {
            str(idx): dataset[idx][1]
            for idx in range(len(dataset))
        }

    assert hasattr(dataset, "class_to_idx"), f"Dataset {name} does not have a class_to_idx attribute."
    assert hasattr(dataset, "file_to_class"), f"Dataset {name} does not have a file_to_class attribute."
    return dataset
