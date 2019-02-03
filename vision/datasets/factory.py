import pretrainedmodels
import pretrainedmodels.utils as utils
from pretrainedmodels import pretrained_settings
from munch import munchify
from bootstrap.lib.options import Options
from .imagenet import Imagenet
from torchvision import transforms

def factory(engine=None):
    dataset = {}

    if Options()['dataset']['name'] == 'imagenet':
        if Options()['dataset']['train_split']:
            dataset['train'] = factory_imagenet(Options()['dataset']['train_split'])

        if Options()['dataset']['eval_split']:
            dataset['eval'] = factory_imagenet(Options()['dataset']['eval_split'])
    else:
        raise ValueError()

    return dataset


def factory_imagenet(split):

    if Options()['dataset']['model']['pretrained']:
        dict_settings = pretrained_settings[Options()['model']['network']['name']]
        dict_settings = dict_settings[Options()['model']['network']['pretrained']]
        model_settings = munchify(dict_settings) # convert a dict into a class (ex: d['mean'] -> d.mean)
        item_tf = utils.TransformImage(model_settings)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        input_dim = Options()['model']['network']['input_dim']
        resize_dim = Options()['model']['network']['resize_dim']

        if split in ['val', 'test']:
            item_tf = transforms.Compose([
                transforms.RandomResizedCrop(input_dim),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            item_tf = transforms.Compose([
                transforms.Resize(resize_dim),
                transforms.CenterCrop(input_dim),
                transforms.ToTensor(),
                normalize,
            ])

    dataset = Imagenet(
        dir_data=Options()['dataset']['dir'],
        split=split,
        batch_size=Options()['dataset']['batch_size'],
        nb_threads=Options()['dataset']['nb_threads'],
        pin_memory=Options()['misc']['cuda'],
        item_tf=item_tf)

    # if Options()['misc'].get('world_size', 1) > 1: # WIP
    #     dataset = torch.utils.data.distributed.DistributedSampler(dataset)

    return dataset

