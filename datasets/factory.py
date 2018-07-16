from bootstrap.lib.options import Options
from .imagenet import Imagenet

import pretrainedmodels
import pretrainedmodels.utils
from munch import munchify

def factory(split):

    dict_settings = pretrainedmodels.pretrained_settings[Options()['model']['network']['name']]
    dict_settings = dict_settings[Options()['model']['network']['pretrained']]
    # convert a dict into a class
    # ex: d['mean'] -> d.mean
    model_settings = munchify(dict_settings)

    if split in ['val', 'test']:
        # - Scale, CenterCrop, ToTensor
        # - ToSpaceBGR (if needed), ToRange255 (if needed)
        # - Normalize(mean, std)
        # https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/utils.py#L31
        item_tf = pretrainedmodels.utils.TransformImage(model_settings)
    else:
        import ipdb; ipdb.set_trace()

    if Options()['dataset']['name'] == 'imagenet':
        dataset = Imagenet(
            dir_data=Options()['dataset']['dir'],
            split=split,
            batch_size=Options()['dataset']['batch_size'],
            nb_threads=Options()['dataset']['nb_threads'],
            pin_memory=Options()['misc']['cuda'],
            item_tf=item_tf)
    else:
        raise ValueError()

    return dataset

