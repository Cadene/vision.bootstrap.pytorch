import torch.nn as nn
import pretrainedmodels
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

def factory(engine=None,):
    Logger()('Creating imagenet network...')
    opt = Options()['model']['network']

    if opt['name'] in pretrainedmodels.model_names:
        network = pretrainedmodels.__dict__[opt['name']](
            num_classes=1000,
            pretrained=opt['pretrained'])

        # takes the first split, be it train, val or test
        split = list(engine.dataset.keys())[0]
        if hasattr(engine.dataset[split], 'classes'):
            nb_classes = len(engine.dataset[split].classes)
            if nb_classes == 2:
                nb_classes = 1 # BCEWithLogitsLoss
            network.last_linear = nn.Linear(network.last_linear.in_features, nb_classes)
        else:
            Logger()('No classes attributs')

        if 'finetuning' in opt and not opt['finetuning']:
            
            for p in network.parameters():
                p.requires_grad = False

            for p in network.last_linear.parameters():
                p.requires_grad = True
    else:
        raise ValueError("--model.network.name not in '{}'".format(pretrainedmodels.model_names))

    return network