import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import load_data

def getdev():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    return dev

class Accuracy:
    """Returns count of correct classes.
    
    Args:
        model_output: tensor of shape [batch size, number of classes], with predicted probability of each class
        target: tensor of shape [batch size], with ground truth class label (index value of the correct class)
    
    Returns:
        torch.Tensor scalar value, with accuracy for the batch
    """
    
    def __init__(self, reduction = None):
        super().__init__()
        self.reduction = reduction

    def __repr__(self):
        return 'Accuracy'

    def __str__(self):
        return 'Accuracy'
    
    def __call__(self, model_output, target_classes):
        _, predicted_class = torch.max(model_output.data, -1)
        if (target_classes.shape != predicted_class.shape):
            print('Warning: predicted_class shape does not match target_class shape')
            print('predicted_class shape:', predicted_class.shape, ' target_classes shape:', target_classes.shape)
        if (self.reduction == 'sum'):
            return (predicted_class == target_classes).sum()
        else:
            return (predicted_class == target_classes).sum() / torch.numel(target_classes)

class Objective:
    """
    Generic class for different types of fitness metrics.
    """
    
    def __init__(self):
        super().__init__()

class NetworkCostObjective(Objective):
    """
    Class for evaluating fitness by cost to build the network.
    """    
    def __init__(self, cost_metric):
        super().__init__()
        self.cost_metric = cost_metric

    def __repr__(self):
        return 'NetworkCostObjective:' + str(self.cost_metric)

    def __call__(self, genome):
        if self.cost_metric == "count":
            return count_connections(genome)
        elif self.cost_metric == "l1":
            return l1_norm(genome)
        elif self.cost_metric == "l2":
            return l2_norm(genome)
        else:
            raise Exception("Unknown network cost metric: ", self.cost_metric)

def count_connections(genome):
    return np.sum([np.count_nonzero(g) for g in genome])

def l1_norm(genome):
    return np.sum([np.sum(np.abs(g)) for g in genome])


def l2_norm(genome):
    return np.sum([np.sum(np.square(g)) for g in genome])

class DatasetObjective(Objective):
    """
    Class for evaluating fitness by performance on a dataset.
    """
    
    def __init__(self, dataset_name, evaluation_metric, dataset_params):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.dataset_params = dataset_params
        
        if dataset_name == 'MNIST':
            train_loader, val_loader, test_loader, input_size = load_data('MNIST', dataset_params)
            self.dataloader = train_loader
        
        if evaluation_metric == 'acc':
            self.metric = Accuracy()

    def __repr__(self):
        return 'DatasetObjective: (' + str(self.dataset_name) + ', ' + str(self.metric) + ') with params:' + str(self.dataset_params)

    def __call__(self, model):
        # We can support evaluating multiple models at the same time in the future
        return evaluate_on_dataset([model], self.dataloader, [self.metric])[0][0]


class FCNet(nn.Module):
    '''
    A simple fully connected model.
    '''
    def __init__(self, layer_size):
        super(FCNet, self).__init__()
        self.layer_size = layer_size

        self.linears = nn.ModuleList([nn.Linear(n_in, n_out) for n_in, n_out in zip(self.layer_size[:-1], self.layer_size[1::])])

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for l in self.linears[:-1]:
            x = F.relu(l(x))
        
        # no relu on last layer
        out = self.linears[-1](x)
        # out = F.relu(self.fc1())
        return out

# Testing functions

def evaluate_on_dataset(evaluation_models, dataloader, metrics, dev = None):
    """Evaluates models on metrics, using the data provided.
    
    Args:
        evaluation_model: list of Pytorch models to evaluate
        dataloader: Pytorch dataloader containing the testing set to use
        metrics: list of metrics to test on
        dev: hardware device to use
    
    Returns:
        tensor containing evaluation for each model for each metric.
    """
    if dev is None:
        dev = getdev()
    if not isinstance(evaluation_models, list):
        evaluation_models = [evaluation_models]
    for m in evaluation_models:
        m.eval()

    loss_eval = np.stack([np.zeros_like(metrics, dtype = float) for m in evaluation_models])
    count_eval = np.stack([np.zeros_like(metrics, dtype = float) for m in evaluation_models])

    with torch.no_grad():
        # loop through batches first, since I think it takes longer to create batches
        for batch_num, (batch_in, batch_target) in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            batch_in, batch_target = batch_in.to(dev), batch_target.to(dev)

            for model_num, model in enumerate(evaluation_models, 0):
                output = model(batch_in)

                for i, m in enumerate(metrics):
                    loss_eval[model_num, i] += m(output, batch_target)
                    count_eval[model_num, i] += np.prod(batch_target.shape)

    loss_eval = loss_eval/count_eval
    return loss_eval

def get_network_shape(genome):
    tmp = [a.shape[0] for a in genome]
    return tuple(tmp[::2] + tmp[-1:])

def evaluate_fitness(genome, objective):
    if len(genome) % 2 != 0:
        raise Exception("Genome format must be [weight matrix, bias vector, weight matrix, bias vector, ...]")

    if isinstance(objective, NetworkCostObjective):
        return objective(genome)

    elif isinstance(objective, DatasetObjective):
        # build the network and run it on the dataset
        dev = getdev()
        net = FCNet(get_network_shape(genome)).to(dev)
        
        sd = net.state_dict()
        
        for i in range(len(net.linears)):
            sd[f'linears.{i}.weight'] = torch.tensor(genome[2*i].T)
            sd[f'linears.{i}.bias'] = torch.tensor(genome[2*i+1])
        
        net.load_state_dict(sd)
        return objective(net)
    else:
        raise Exception("Unknown objective type:", objective)