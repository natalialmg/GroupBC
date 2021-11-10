
import sys
sys.path.append("../")
import argparse
import pandas as pd
from distutils.util import strtobool
from general.dataloaders import *
from general.datasets import *
from models.robust_training import euclidean_proj_simplex
from general.network import *
from general.utils import *
from models.training import *
import torch
from models.model import *
from general.network import *

from opacus.utils import module_modification
import numpy as np

cparser = argparse.ArgumentParser()

## General specs
cparser.add_argument('--gpu', action='store', default=0, type=int,help='gpu')
cparser.add_argument('--model_name', action='store', default= 'vanilla_resnet18', type=str, help='model_name')
cparser.add_argument('--basedir', action='store', default= '/data/natalia/models/', type=str,help='basedir for internal model save')
cparser.add_argument('--seed', action='store', default=42, type=int, help='randomizer seed')
cparser.add_argument('--seed_dataset', action='store', default=42, type=int, help='randomizer seed dataset')
cparser.add_argument('--split', action='store', default=1, type=int, help='split dataset, int > 0')
cparser.add_argument('--loadmodel', action='store', default= '', type=str, help='model_dir')
cparser.add_argument('--dataset_reduction', action='store', default=1.0, type=float, help='fraction of training dataset to use ')

## Dataset/ Dataloader
cparser.add_argument('--dataset', action='store', default='cifar100_coarse', type=str,help='dataset')
cparser.add_argument('--network', action='store', default='resnet18', type=str,help='network')
cparser.add_argument('--pretrained', action='store', default=True, type=lambda x: bool(strtobool(x)), help='boolean: pretrained network (default true)')
cparser.add_argument('--batch', action='store', default=64, type=int, help='batch size')
cparser.add_argument('--augmentation', action='store', default=True, type=lambda x: bool(strtobool(x)), help='boolean: data augmentation (default true)')
cparser.add_argument('--normlayer', action='store', default='batchnorm', type=str,help='normalization layer, default is batchnorm')

## Model Learner
cparser.add_argument('--epochs', action='store', default=501, type=int, help='epochs')
cparser.add_argument('--epochs_warmup', action='store', default=1000, type=int, help='epochs warmup')
cparser.add_argument('--loss', action='store', default='CE', type=str,help='loss')
cparser.add_argument('--train_mode', action='store', default='erm', type=str,  help='string: train mode (default vanilla)')

cparser.add_argument('--lr', action='store', default=1e-4, type=float, help='learners learning rate ')
cparser.add_argument('--optim', action='store', default='adam', type=str,help='Learners optimizer')
cparser.add_argument('--regression', action='store', default=False,type=lambda x: bool(strtobool(x)),help='boolean: regression (default false)')
cparser.add_argument('--optim_wreg', action='store', default=0, type=float, help='weight (l2) reg in optimizer')
cparser.add_argument('--scheduler', action='store', default=None, type=str,help='scheduler: MultiStepLR, OneCycleLR, CosineAnnealingLR')
cparser.add_argument('--patience', action='store', default=0, type=int, help='patience of stopper criteria, 0 means ignore patience')


## Robust learners:
cparser.add_argument('--max_weight_change', action='store', default=0.25, type=float, help='max_weight_change')
cparser.add_argument('--cost_delta_improve', action='store', default=0.25, type=float, help='max cost_delta_improve')
cparser.add_argument('--lr_weight', action='store', default=0.1, type=float, help='lr weight PGA ')
cparser.add_argument('--min_weight', action='store', default=0.0, type=float, help='min group weight')
cparser.add_argument('--min_weight_prior', action='store', default=False,type=lambda x: bool(strtobool(x)),help='boolean: min weight = cparser.min_weight * group_prior')


cparser = cparser.parse_args()


# datasets_included = ['ham10000','cifar100', 'celeba' ]

if __name__== '__main__':

    ##### DATASETS
    num_max_batch = np.ceil(5000 / cparser.batch) #number of batches to consider as one epoch

    if cparser.dataset == 'cifar10':
        # only default split in terms of test train
        pd_train, pd_test = CIFAR10()
        file_tag = 'data'
        strat_tag = 'utility'
        group_tag_dataset = 'utility'
        convert_conv1 = True  ## convert to kernel 3 convolution first layer resnet

        ## CIFAR Transformations
        import torchvision.transforms as tt

        stats = ((0.5074, 0.4867, 0.4411),
                 (0.2011, 0.1987, 0.2025))

        train_transform = tt.Compose([
            tt.ToPILImage(),
            # tt.RandomCrop(32, padding=4, padding_mode='reflect'),
            tt.RandomCrop(32, 4),
            tt.RandomHorizontalFlip(),
            # tt.RandomVerticalFlip(),
            tt.ToTensor(),
            tt.Normalize(*stats)
        ])

        test_transform = tt.Compose([
            tt.ToPILImage(),
            tt.ToTensor(),
            tt.Normalize(*stats)
        ])

    if cparser.dataset == 'celebA_blond':
        # only default split in terms of test train
        pd_train, pd_test = CelebA(utility="Blond_Hair")  #, split=1, n_splits=5, seed=42)
        file_tag = 'filepath'
        strat_tag = 'utility'
        group_tag_dataset = 'group'
        convert_conv1 = False
        num_max_batch = np.ceil(10000/cparser.batch) #number of batches to consider as one epoch

        import torchvision.transforms as tt
        stats = ((0.485, 0.456, 0.406),
                 (0.229, 0.224, 0.225))
        if cparser.augmentation:
            train_transform = tt.Compose([
                tt.CenterCrop((178, 178)),
                tt.Resize((128, 128)),
                # tt.Resize(64),
                tt.RandomHorizontalFlip(),
                tt.ToTensor(),
                tt.Normalize(*stats)
            ])
        else:
            train_transform = tt.Compose([
            tt.CenterCrop((178, 178)),
            tt.Resize((128, 128)),
            # tt.Resize(64),
            tt.ToTensor(),
            tt.Normalize(*stats)
            ])

        test_transform = tt.Compose([
            tt.CenterCrop((178, 178)),
            tt.Resize((128, 128)),
            # tt.Resize(64),
            tt.ToTensor(),
            tt.Normalize(*stats)
            ])

    if cparser.dataset == 'CUB':
        # only default split in terms of test train
        pd_train, pd_test = CUB(utility="y")
        file_tag = 'filepath'
        strat_tag = 'utility'
        group_tag_dataset = 'group'
        convert_conv1 = False

        import torchvision.transforms as tt

        stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        scale = 256.0 / 128.0
        target_resolution = (128, 128)

        # scale = 256.0 / 224.0
        # target_resolution = (224, 224)

        if cparser.augmentation:
            train_transform = tt.Compose([
                tt.Resize((int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
                tt.CenterCrop(target_resolution),
                tt.RandomHorizontalFlip(),
                tt.ToTensor(),
                tt.Normalize(*stats)
            ])
        else:
            train_transform = tt.Compose([
            tt.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            tt.CenterCrop(target_resolution),
            tt.ToTensor(),
            tt.Normalize(*stats)
            ])

        test_transform = tt.Compose([
            # tt.ToPILImage(),
            tt.Resize((int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
            tt.CenterCrop(target_resolution),
            tt.ToTensor(),
            tt.Normalize(*stats)
        ])

    if cparser.split > 0:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, random_state=cparser.seed_dataset)
        skf.get_n_splits(pd_train)
        index_splits = skf.split(pd_train, pd_train[strat_tag].values)

        ix = 1
        cparser.split = np.minimum(cparser.split,5)
        for train_index, test_index in index_splits:
            pd_train_split, pd_val_split = pd_train.iloc[train_index], pd_train.iloc[test_index]
            if ix == cparser.split:
                print("split:", ix)
                print("test index :", test_index,"; train index :", train_index)
                break
    else:
        ## split <= 0 means no split
        pd_train_split = pd_train.copy()
        pd_val_split = pd_test.copy()

    pd_train_split['dataset'] = 'train'
    pd_val_split['dataset'] = 'validation'
    pd_test['dataset'] = 'test'

    #### training/val dataset reduction (optional)
    pd_all = pd.DataFrame()
    if cparser.dataset_reduction < 1.0:
        from sklearn.model_selection import train_test_split

        #train set reduction
        ixg0, ixg1 = train_test_split(np.arange(len(pd_train_split)),
                                      train_size=cparser.dataset_reduction,
                                      random_state=cparser.seed_dataset,
                                      stratify=pd_train_split[strat_tag].values)

        pd_other = pd_train_split.iloc[ixg1]
        pd_train_split = pd_train_split.iloc[ixg0]

        #val set reduction
        ixg0, ixg1 = train_test_split(np.arange(len(pd_val_split)),
                                      train_size=cparser.dataset_reduction,
                                      random_state=cparser.seed_dataset,
                                      stratify=pd_val_split[strat_tag].values)

        pd_other = pd.concat([pd_other, pd_val_split.iloc[ixg1]])
        pd_other['dataset'] = 'other'
        pd_val_split = pd_val_split.iloc[ixg0]

        pd_all = pd.concat([pd_all, pd_other], axis=0)

    pd_all = pd.concat([pd_all, pd_train_split], axis=0)
    pd_all = pd.concat([pd_all, pd_val_split], axis=0)
    pd_all = pd.concat([pd_all, pd_test], axis=0)

    print('Train split : ')
    print(pd_train_split.groupby(group_tag_dataset)['dataset'].count())
    print(pd_train_split.groupby(group_tag_dataset)['dataset'].count()/len(pd_train_split))

    print('Val split : ')
    print(pd_val_split.groupby(group_tag_dataset)['dataset'].count())
    print(pd_val_split.groupby(group_tag_dataset)['dataset'].count()/len(pd_val_split))

    print('Test split : ')
    print(pd_test.groupby(group_tag_dataset)['dataset'].count())
    print(pd_test.groupby(group_tag_dataset)['dataset'].count()/len(pd_test))


    ######## CONFIG ########
    nutility = len(pd_train['utility'].unique())

    network_params = {'network': cparser.network,
                      'normlayer': cparser.normlayer,
                      'pretrained': cparser.pretrained}

    optimizer_params = {'optimizer': cparser.optim,
                        'LEARNING_RATE': cparser.lr,
                        'weight_decay': cparser.optim_wreg}

    if cparser.patience == 0: #patience for stopping criteria, zero means should train the selected epochs
        cparser.patience = cparser.epochs + 2

    ### Default parameters for all possible schedulers
    scheduler_params = {'steps_per_epoch': int(np.ceil(len(pd_train_split)/cparser.batch)),
                        'T_max': 200,
                        'max_lr': cparser.lr,
                        'milestones':[100,150],
                        'gamma':0.1,
                        'lr_decay':0.9,
                        'patience':5} #patience for ManualLRDecayPlateau

    train_mode_params = {}

    ### Dataloaders ###
    sampler_on = False
    if cparser.train_mode in ['gmmf' ,'gb'] : ## config for group modalities ('gmmf = group minmax fairness, 'gb': group balance)

        group_tag = group_tag_dataset
        group_prior = pd_train_split.groupby(group_tag)['utility'].count().values / len(pd_train_split)

        train_mode_params['group_constrain'] = np.zeros_like(group_prior)
        train_mode_params['group_constrain_acc'] = np.ones_like(group_prior)
        train_mode_params['group_prior'] = group_prior

        train_mode_params['max_weight_change'] = cparser.max_weight_change
        train_mode_params['cost_delta_improve'] = cparser.cost_delta_improve
        if cparser.min_weight_prior :
            train_mode_params['min_weight'] = cparser.min_weight*group_prior
        else:
            train_mode_params['min_weight'] = cparser.min_weight*np.ones_like(group_prior)
        train_mode_params['lr_penalty'] = cparser.lr_weight

        # print(train_mode_params['min_weight'], np.sum(train_mode_params['min_weight']))
        if np.sum(train_mode_params['min_weight']) > 1:
            train_mode_params['min_weight'] = train_mode_params['min_weight']/np.sum(train_mode_params['min_weight'])

        weight_init = to_np(euclidean_proj_simplex(torch.from_numpy(group_prior), s=1, e=train_mode_params['min_weight']))
        train_mode_params['weights_init'] = weight_init


        if cparser.train_mode == 'gb':
            train_mode_params['lr_penalty'] = 0.0
            pd_train_split['weights_sampler'] = [1 / group_prior[g] for g in pd_train_split[
                group_tag].values]  # balance sampler
            sampler_on = True
            print('group_reweight_sampler : ', pd_train_split.groupby(group_tag).mean())

        print('group_prior :', train_mode_params['group_prior'] )
        print('group_constrain :',train_mode_params['group_constrain'] )
        print('group_constrain_acc :',train_mode_params['group_constrain_acc'] )

        print('min weight :', train_mode_params['min_weight'])
        print('weight_init :', train_mode_params['weights_init'])
        group2cat = True

    else:
        group_tag = None
        group2cat = False


    config = Config(n_utility=nutility,
                    basedir=cparser.basedir, model_name=cparser.model_name, seed=cparser.seed,
                    BATCH_SIZE=cparser.batch, type_loss = cparser.loss, type_metric = ['acc'],
                    GPU_ID = cparser.gpu,
                    network_params=network_params,
                    optimizer_params=optimizer_params,
                    scheduler=cparser.scheduler,
                    scheduler_params=scheduler_params,
                    patience=cparser.patience, num_max_batch=num_max_batch,
                    EPOCHS=cparser.epochs,epochs_warmup=cparser.epochs_warmup,
                    train_mode = cparser.train_mode,train_mode_params=train_mode_params,
                    regression=cparser.regression
                    )

    config.save_json()


    ### Dataloaders ###
    train_dataloader = get_dataloaders_image(pd_train_split, file_tag=file_tag, utility_tag='utility',
                                             group_tag=group_tag,
                                             augmentations=train_transform, shuffle=True,
                                             num_workers=8, batch_size=config.BATCH_SIZE,group2cat=group2cat,
                                             sampler_on = sampler_on)

    val_dataloader = get_dataloaders_image(pd_val_split, file_tag=file_tag, utility_tag='utility',
                                           group_tag=group_tag,
                                           augmentations=test_transform, shuffle=False,
                                           num_workers=8, batch_size=config.BATCH_SIZE,group2cat=group2cat)

    eval_dataloader = get_dataloaders_image(pd_all, file_tag=file_tag, utility_tag='utility',
                                            group_tag=group_tag,
                                            augmentations=test_transform, shuffle=False,
                                            num_workers=8, batch_size=config.BATCH_SIZE,group2cat=group2cat)

    #### Classifier ####

    classifier_network = get_resnet(config.n_utility, pretrained = config.network_params['pretrained'],
                                    typenet = config.network_params['network'],convert_conv1=convert_conv1)

    if cparser.loadmodel != '':
        if os.path.exists(cparser.loadmodel):
            print(' Loading : ',cparser.loadmodel )
            model_params_load(cparser.loadmodel,classifier_network,None,config.DEVICE)

    if cparser.normlayer == 'groupnorm':
        print('BN to GN')
        classifier_network = module_modification.convert_batchnorm_modules(classifier_network)
    elif cparser.normlayer == 'identity':
        print('BN to identity')
        classifier_network = module_modification.convert_batchnorm_modules(classifier_network,
                                                                           module_modification.nullify_batchnorm_modules)
    elif cparser.normlayer == 'instance':
        print('BN to instance norm')
        classifier_network = module_modification.convert_batchnorm_modules(classifier_network,
                                                                           module_modification._batchnorm_to_instancenorm)


    ### Trainer ###
    trainer = Model(config, train_dataloader, val_dataloader, classifier_network)


    ## Train ###
    history, evaluation_list = trainer.train_model(epochs=config.EPOCHS,
                                                   epoch_warmup=config.epochs_warmup,
                                                   eval_dataloader=eval_dataloader,
                                                   metric_stopper=config.type_metric[0],
                                                   train_modality = config.train_mode)

    # cast all np arrays in history to list
    for key in history.keys():
        history[key] = np.array(history[key]).tolist()

    # save history & evaluation
    save_json(history, trainer.config.basedir + trainer.config.model_name + '/history.json')
    print('history file saved on : ', trainer.config.basedir + trainer.config.model_name + '/history.json')

    for i in range(len(evaluation_list)):
        evaluation_list[i]['dataset'] = pd_all['dataset'].values
        evaluation_list[i]['sample_index'] = pd_all['sample_index'].values
        evaluation_list[i]['group_tag'] = pd_all[group_tag_dataset].values

        evaluation_list[i].to_csv(trainer.config.basedir + trainer.config.model_name + '/eval' + str(i) + '.csv', index=0)
        print('Saving : ', trainer.config.basedir + trainer.config.model_name + '/eval' + str(i) + '.csv' )











