import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from .augmentations import RandAugment, CutoutRandom
import pickle
import os


def get_cifar10(root='data/datasets', n_lbl=4000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt=''):
    os.makedirs(root, exist_ok=True) #create the root directory for saving data
    # augmentations
    transform_train = transforms.Compose([
        RandAugment(3,4),  #from https://arxiv.org/pdf/1909.13719.pdf. For CIFAR-10 M=3, N=4
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
        CutoutRandom(n_holes=1, length=16, random=True)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
    ])

    if ssl_idx is None:
        base_dataset = datasets.CIFAR10(root, train=True, download=True)
        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.targets, n_lbl, 10)
        
        os.makedirs('data/splits', exist_ok=True)
        f = open(os.path.join('data/splits', f'cifar10_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict,f)
    
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']

    lbl_idx = train_lbl_idx
    if pseudo_lbl is not None:
        pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']
        lbl_idx = np.array(lbl_idx + pseudo_idx)

        #balance the labeled and unlabeled data 
        if len(nl_idx) > len(lbl_idx):
            exapand_labeled = len(nl_idx) // len(lbl_idx)
            lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

            if len(lbl_idx) < len(nl_idx):
                diff = len(nl_idx) - len(lbl_idx)
                lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
            else:
                assert len(lbl_idx) == len(nl_idx)
    else:
        pseudo_idx = None
        pseudo_target = None
        nl_idx = None
        nl_mask = None

    train_lbl_dataset = CIFAR10SSL(
        root, lbl_idx, train=True, transform=transform_train,
        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
        nl_idx=nl_idx, nl_mask=nl_mask)
    
    if nl_idx is not None:
        train_nl_dataset = CIFAR10SSL(
            root, np.array(nl_idx), train=True, transform=transform_train,
            pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
            nl_idx=nl_idx, nl_mask=nl_mask)

    train_unlbl_dataset = CIFAR10SSL(
    root, train_unlbl_idx, train=True, transform=transform_val)

    test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=True)

    if nl_idx is not None:
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, test_dataset
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, test_dataset


def get_cifar100(root='data/datasets', n_lbl=10000, ssl_idx=None, pseudo_lbl=None, itr=0, split_txt=''):
    ## augmentations
    transform_train = transforms.Compose([
        RandAugment(3,4),  #from https://arxiv.org/pdf/1909.13719.pdf. For CIFAR-10 M=3, N=4
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
        CutoutRandom(n_holes=1, length=16, random=True)
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    if ssl_idx is None:
        base_dataset = datasets.CIFAR100(root, train=True, download=True)
        train_lbl_idx, train_unlbl_idx = lbl_unlbl_split(base_dataset.targets, n_lbl, 100)
        
        f = open(os.path.join('data/splits', f'cifar100_basesplit_{n_lbl}_{split_txt}.pkl'),"wb")
        lbl_unlbl_dict = {'lbl_idx': train_lbl_idx, 'unlbl_idx': train_unlbl_idx}
        pickle.dump(lbl_unlbl_dict,f)
    
    else:
        lbl_unlbl_dict = pickle.load(open(ssl_idx, 'rb'))
        train_lbl_idx = lbl_unlbl_dict['lbl_idx']
        train_unlbl_idx = lbl_unlbl_dict['unlbl_idx']

    lbl_idx = train_lbl_idx
    if pseudo_lbl is not None:
        pseudo_lbl_dict = pickle.load(open(pseudo_lbl, 'rb'))
        pseudo_idx = pseudo_lbl_dict['pseudo_idx']
        pseudo_target = pseudo_lbl_dict['pseudo_target']
        nl_idx = pseudo_lbl_dict['nl_idx']
        nl_mask = pseudo_lbl_dict['nl_mask']
        lbl_idx = np.array(lbl_idx + pseudo_idx)

        #balance the labeled and unlabeled data 
        if len(nl_idx) > len(lbl_idx):
            exapand_labeled = len(nl_idx) // len(lbl_idx)
            lbl_idx = np.hstack([lbl_idx for _ in range(exapand_labeled)])

            if len(lbl_idx) < len(nl_idx):
                diff = len(nl_idx) - len(lbl_idx)
                lbl_idx = np.hstack((lbl_idx, np.random.choice(lbl_idx, diff)))
            else:
                assert len(lbl_idx) == len(nl_idx)
    else:
        pseudo_idx = None
        pseudo_target = None
        nl_idx = None
        nl_mask = None

    train_lbl_dataset = CIFAR100SSL(
        root, lbl_idx, train=True, transform=transform_train,
        pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
        nl_idx=nl_idx, nl_mask=nl_mask)
    
    if nl_idx is not None:
        train_nl_dataset = CIFAR100SSL(
            root, np.array(nl_idx), train=True, transform=transform_train,
            pseudo_idx=pseudo_idx, pseudo_target=pseudo_target,
            nl_idx=nl_idx, nl_mask=nl_mask)

    train_unlbl_dataset = CIFAR100SSL(
    root, train_unlbl_idx, train=True, transform=transform_val)

    test_dataset = datasets.CIFAR100(root, train=False, transform=transform_val, download=True)

    if nl_idx is not None:
        return train_lbl_dataset, train_nl_dataset, train_unlbl_dataset, test_dataset
    else:
        return train_lbl_dataset, train_unlbl_dataset, train_unlbl_dataset, test_dataset


def lbl_unlbl_split(lbls, n_lbl, n_class):
    lbl_per_class = n_lbl // n_class
    lbls = np.array(lbls)
    lbl_idx = []
    unlbl_idx = []
    for i in range(n_class):
        idx = np.where(lbls == i)[0]
        np.random.shuffle(idx)
        lbl_idx.extend(idx[:lbl_per_class])
        unlbl_idx.extend(idx[lbl_per_class:])
    return lbl_idx, unlbl_idx


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        
        self.targets = np.array(self.targets)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))
        
        if nl_mask is not None:
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index], self.nl_mask[index]


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        
        self.targets = np.array(self.targets)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))
        
        if nl_mask is not None:
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
        

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target, self.indexs[index], self.nl_mask[index]