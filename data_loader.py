import torch
import torchvision.datasets as dsets
from torchvision import transforms
import pathlib
import urllib
import shutil
import os
import pathlib
import zipfile
import tarfile


class DataLoader():
    def __init__(self, dataset, image_path, image_size, batch_size, center_corp=0, shuf=True):
        self.dataset = dataset
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuf = shuf
        self.center_corp = center_corp

    def download_data(self, out_path, url, extract=True, force=False):
        pathlib.Path(out_path).mkdir(exist_ok=True)
        out_filename = os.path.join(out_path, os.path.basename(url))

        if os.path.isfile(out_filename) and not force:
            print(f'File {out_filename} exists, skipping download.')
        else:
            print(f'Downloading {url}...')

            with urllib.request.urlopen(url) as response:
                with open(out_filename, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)

            print(f'Saved to {out_filename}.')

        extracted_dir = None
        if extract and out_filename.endswith('.zip'):
            print(f'Extracting {out_filename}...')
            with zipfile.ZipFile(out_filename, "r") as zipf:
                names = zipf.namelist()
                zipf.extractall(out_path)
                zipinfos = zipf.infolist()
                first_dir = next(filter(lambda zi: zi.is_dir(), zipinfos)).filename
                extracted_dir = os.path.join(out_path, os.path.dirname(first_dir))
                print(f'Extracted {len(names)} to {extracted_dir}')

        if extract and out_filename.endswith(('.tar.gz', '.tgz')):
            print(f'Extracting {out_filename}...')
            with tarfile.TarFile(out_filename, "r") as tarf:
                members = tarf.getmembers()
                tarf.extractall(out_path)
                first_dir = next(filter(lambda ti: ti.isdir(), members)).name
                extracted_dir = os.path.join(out_path, os.path.dirname(first_dir))
                print(f'Extracted {len(members)} to {extracted_dir}')

        return out_filename, extracted_dir

    def transform(self, resize, totensor, normalize):
        options = []
        if self.center_corp > 0:
            options.append(transforms.CenterCrop(self.center_corp))
            #options.append(transforms.CenterCrop(self.imsize))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, True)
        dataset = dsets.LSUN(self.path + '/lsun', classes=[classes], transform=transforms)
        return dataset, 0

    def load_celeba(self):
        transforms = self.transform(True, True, True)  #, True)
        #dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        dataset = dsets.CelebA(self.path+'/CelebA', transform=transforms) #, split='all')
        return dataset, 0

    def load_cifar(self):
        transforms = self.transform(False, True, True)  #, False)
        cifar10_train_ds = dsets.CIFAR10(root=self.path+'/cifar-10/', download=True, train=True,
                                         transform=transforms)

        print('Number of samples:', len(cifar10_train_ds))
        return cifar10_train_ds, 10

    def load_gwb(self):
        DATA_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-bush.zip'
        _, dataset_dir = self.download_data(out_path=self.path+'/gwb', url=DATA_URL, extract=True, force=False)
        transforms = self.transform(True, True, True)  #, False)

        ds_gwb = dsets.ImageFolder(os.path.dirname(dataset_dir), transforms)
        return ds_gwb, 0

    def load(self):
        if self.dataset == 'lsun':
            dataset, n_classes = self.load_lsun()
        elif self.dataset == 'celeba':
            dataset, n_classes = self.load_celeba()
        elif self.dataset == 'cifar':
            dataset, n_classes = self.load_cifar()
        elif self.dataset == 'gwb':
            dataset, n_classes = self.load_gwb()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=self.shuf)
        return loader, n_classes

