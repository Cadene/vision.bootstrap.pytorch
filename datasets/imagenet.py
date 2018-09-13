import os
import torch
import torch.utils.data as data

from torchvision import datasets
from torchvision import transforms

from bootstrap.datasets import transforms as bootstrap_tf

class Imagenet(data.Dataset):

    def __init__(self,
                 dir_data='data/imagenet',
                 split='train',
                 batch_size=100,
                 nb_threads=1,
                 pin_memory=True,
                 item_tf=None):
        self.dir_data = dir_data
        self.split = split
        self.batch_size = batch_size
        self.nb_threads = nb_threads
        self.pin_memory = pin_memory
        self.item_tf = item_tf

        self.dir_split = os.path.join(self.dir_data, self.split)

        if self.split == 'train':
            self.shuffle = True
        elif self.split in ['val', 'test']:
            self.shuffle = False
        else:
            raise ValueError()

        self.dataset = datasets.ImageFolder(self.dir_split, self.item_tf)

        # the usual collate function for bootstrap
        # to handle the (potentially nested) dict item format bellow
        self.collate_fn = bootstrap_tf.Compose([
            bootstrap_tf.ListDictsToDictLists(),
            bootstrap_tf.StackTensors()
        ])
        
    def __getitem__(self, index):
        data, class_id = self.dataset[index]
        item = {}
        item['index'] = index
        item['data'] = data
        item['class_id'] = torch.LongTensor([class_id])
        return item

    def __len__(self):
        return len(self.dataset)

    def make_batch_loader(self):
        data_loader = data.DataLoader(self,
            batch_size=self.batch_size,
            num_workers=self.nb_threads,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=False)
        return data_loader