import torch.nn as nn
import pretrainedmodels
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger

class Wrapper(nn.Module):

    def __init__(self, network):
        super(Wrapper, self).__init__()
        self.network = network

    def forward(self, batch):
        out = self.network(batch['data'])
        return out

    def features(self, batch):
        out = self.network.features(batch['data'])
        return out


def factory(engine=None, opt=None):
    if opt is None:
        opt = Options()['model']['network']

    Logger()('Creating imagenet network...')

    if opt['name'] in pretrainedmodels.model_names:
        network = pretrainedmodels.__dict__[opt['name']](
            num_classes=1000,
            pretrained=opt['pretrained'])

        # takes the first split, be it train, val or test
        split = list(engine.dataset.keys())[0]
        if hasattr(engine.dataset[split], 'classes'):
            nb_classes = len(engine.dataset[split].classes)    
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

    network = Wrapper(network)
    return network