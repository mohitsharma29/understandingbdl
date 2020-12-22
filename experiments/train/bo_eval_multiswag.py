import argparse
import os, sys 
import time
import tabulate

sys.path.append('/home/mohit/understandingbdl/')
import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from swag import data, models, utils, losses
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--savedir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--swag_ckpts', type=str, nargs='*', required=True, 
                    help='list of SWAG checkpoints')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='model',
                    help='model name (default: none)')
parser.add_argument('--label_arr', default=None, help="shuffled label array")

parser.add_argument('--max_num_models', type=int, default=20, help='maximum number of SWAG models to save')
parser.add_argument('--swag_samples', type=int, default=20, metavar='N', 
                    help='number of samples from each SWAG model (default: 20)')
# New Project parameter
parser.add_argument('--warm_start', type=bool, default=False, help='warm start for BO with K-means')
parser.add_argument('--warm_k', type=int, default=1, help='Value of k for kmeans during warm start')

args = parser.parse_args()
args.inference = 'low_rank_gaussian'
args.subspace = 'covariance'
args.no_cov_mat = False


args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

#args.device = torch.device('cpu')

torch.backends.cudnn.benchmark = True
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    split_classes=None
    )

print('Using Noisy Test set')
corrupted_testset = np.load("/media/data_dump/Mohit/bayesianML/cifar-corrupted/motion_blur_5.npz")
loaders['test'].dataset.data = corrupted_testset["data"]
loaders['test'].dataset.targets = corrupted_testset["labels"]

if args.label_arr:
    print("Using labels from {}".format(args.label_arr))
    label_arr = np.load(args.label_arr)
    print("Corruption:", (loaders['train'].dataset.targets != label_arr).mean())
    loaders['train'].dataset.targets = label_arr

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes,
                       **model_cfg.kwargs)
model.to(args.device)
print("Model has {} parameters".format(sum([p.numel() for p in model.parameters()])))


swag_model = SWAG(model_cfg.base,
                args.subspace, {'max_rank': args.max_num_models},
                *model_cfg.args, num_classes=num_classes,
                **model_cfg.kwargs)
swag_model.to(args.device)


columns = ['swag', 'sample', 'te_loss', 'te_acc', 'ens_loss', 'ens_acc']

if args.warm_start:
    print('estimating initial PI using K-Means')
    kmeans_input = []
    for ckpt_i, ckpt in enumerate(args.swag_ckpts):
        #print("Checkpoint {}".format(ckpt))
        checkpoint = torch.load(ckpt)
        swag_model.subspace.rank = torch.tensor(0)
        swag_model.load_state_dict(checkpoint['state_dict'])
        mean, variance = swag_model._get_mean_and_variance()
        kmeans_input.append({
            'model':ckpt,
            'mean': mean,
            'variance': variance})

paramDump = []
def boSwag(Pi):
    useMetric='nll'
    disableBo=False
    if disableBo==True:
        print("Computing for base case")
        # Base case = uniform weights for each SWAG and its models
        swagCount = 0
        for ckpt_i, ckpt in enumerate(args.swag_ckpts):
            swagCount += 1
        n_ensembled = 0.
        multiswag_probs = None
        for ckpt_i, ckpt in enumerate(args.swag_ckpts):
            print("Checkpoint {}".format(ckpt))
            checkpoint = torch.load(ckpt)
            swag_model.subspace.rank = torch.tensor(0)
            swag_model.load_state_dict(checkpoint['state_dict'])

            for sample in range(args.swag_samples):
                swag_model.sample(.5)
                utils.bn_update(loaders['train'], swag_model)
                res = utils.predict(loaders['test'], swag_model)
                probs = res['predictions']
                targets = res['targets']
                nll = utils.nll(probs, targets)
                acc = utils.accuracy(probs, targets)

                if multiswag_probs is None:
                    multiswag_probs = probs.copy()
                else:
                    #TODO: rewrite in a numerically stable way
                    multiswag_probs +=  (probs - multiswag_probs)/ (n_ensembled + 1)
                n_ensembled += 1

                ens_nll = utils.nll(multiswag_probs, targets)
                ens_acc = utils.accuracy(multiswag_probs, targets)
                values = [ckpt_i, sample, nll, acc, ens_nll, ens_acc]
                table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
                print(table)
        initialPi = [1/swagCount]*swagCount
        return (initialPi, ens_nll, ens_acc)
    else:
        n_ensembled = 0.
        multiswag_probs = None
        for ckpt_i, ckpt in enumerate(args.swag_ckpts):
            #print("Checkpoint {}".format(ckpt))
            checkpoint = torch.load(ckpt)
            swag_model.subspace.rank = torch.tensor(0)
            swag_model.load_state_dict(checkpoint['state_dict'])
            #swagWeight = Pi[ckpt]
            #swagWeight = Pi[ckpt]/sum([Pi[i] for i in Pi])
            swagWeight = Pi[ckpt_i]/sum(Pi)
            indivWeight = swagWeight/args.swag_samples

            for sample in range(args.swag_samples):
                swag_model.sample(.5)
                utils.bn_update(loaders['train'], swag_model)
                res = utils.predict(loaders['test'], swag_model)
                probs = res['predictions']
                targets = res['targets']
                nll = utils.nll(probs, targets)
                acc = utils.accuracy(probs, targets)

                if multiswag_probs is None:
                    multiswag_probs = indivWeight*probs.copy()
                else:
                    #TODO: rewrite in a numerically stable way
                    #multiswag_probs +=  (probs - multiswag_probs)/ (n_ensembled + 1)
                    multiswag_probs += indivWeight*probs.copy()
                n_ensembled += 1

                ens_nll = utils.nll(multiswag_probs, targets)
                ens_acc = utils.accuracy(multiswag_probs, targets)
                values = [ckpt_i, sample, nll, acc, ens_nll, ens_acc]
                #table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
                #print(table)
        paramDump.append([Pi,ens_nll, ens_acc])
        print(Pi, ens_nll, ens_acc)
        if useMetric == 'nll':
            return ens_nll
        else:
            return ens_acc

import numpy as np
import tqdm
import json
results = []
import time

with open('swagModelPaths.txt', 'r') as f:
    swag_ckpts = f.readlines()[0][1:-1].split(' ')

for i in range(20):
    print('Starting Iteration ' + str(i))
    #start = time.time()
    initialGuess = np.random.dirichlet(np.ones(len(swag_ckpts)),size=1)[0]
    #initialGuess = initialGuess/sum(initialGuess)
    tempNLL = boSwag(initialGuess)
    results.append([initialGuess, tempNLL])
    #stop = time.time()
    #print(stop-start)

np.save('dirichletRandomGuess.npy', results)

"""
Pi = {}
for ckpt_i, ckpt in enumerate(args.swag_ckpts):
    Pi[ckpt] = 0

from ax import (
    ComparisonOp,
    ParameterType, 
    RangeParameter,
    SearchSpace, 
    SimpleExperiment, 
    OutcomeConstraint,
    SumConstraint
)
from ax.modelbridge.registry import Models

parameters = []
for i in Pi:
    parameters.append(RangeParameter(name=i, parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0))
#probCons = SumConstraint(parameters=[i for i in parameters], is_upper_bound=True, bound=1.0)

#swagSearchSpace = SearchSpace(parameters=parameters, parameter_constraints=[probCons])
swagSearchSpace = SearchSpace(parameters=parameters)

exp = SimpleExperiment(
    name="BO_SWAG",
    search_space=swagSearchSpace,
    evaluation_function=boSwag,
    objective_name="testSwag",
    minimize=True,
)

print(f"Running Sobol initialization trials...")
sobol = Models.SOBOL(exp.search_space)
for i in range(5):
    exp.new_trial(generator_run=sobol.gen(1))
    
for i in range(25):
    print(f"Running GP+EI optimization trial {i+1}/25...")
    # Reinitialize GP+EI model at each step with updated data.
    gpei = Models.BOTORCH(experiment=exp, data=exp.eval())
    batch = exp.new_trial(generator_run=gpei.gen(1))

print(exp.eval_trial(exp.trials[1]))
print("Done!")"""

"""
print('Preparing directory %s' % args.savedir)
os.makedirs(args.savedir, exist_ok=True)
with open(os.path.join(args.savedir, 'eval_command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

np.savez(os.path.join(args.savedir, "multiswag_probs.npz"),
         predictions=multiswag_probs,
         targets=targets)"""
