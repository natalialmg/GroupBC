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
cparser.add_argument('--addclass', action='store', default=0, type=int, help='split dataset, int > 0')
cparser.add_argument('--previous_model', action='store', default= '', type=str, help='previous model directory path')

cparser.add_argument('--gpu', action='store', default=0, type=int,help='gpu')
cparser.add_argument('--model_name', action='store', default= 'vanilla_resnet18', type=str, help='model_name')
cparser.add_argument('--basedir', action='store', default= '/data/natalia/models/', type=str,help='basedir for internal model save')
cparser.add_argument('--seed', action='store', default=42, type=int, help='randomizer seed')
cparser.add_argument('--seed_dataset', action='store', default=42, type=int, help='randomizer seed dataset')
cparser.add_argument('--split', action='store', default=1, type=int, help='split dataset, int > 0')
cparser.add_argument('--loadmodel', action='store', default= '', type=str, help='model_dir')

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

    ######################################
    # Previous model predictions
    ######################################

    ## Load previous model
    config_json = load_json(cparser.previous_model + '/config.json')
    config_json['GPU_ID'] = cparser.gpu
    config = Config(config_dic=config_json)
    print(config)
    from opacus.utils import module_modification

    classifier_network = get_resnet(config.n_utility, pretrained=cparser.pretrained,
                                    typenet=cparser.network, convert_conv1=convert_conv1)

    if config.network_params['normlayer'] == 'groupnorm':
        print('BN to GN')
        classifier_network = module_modification.convert_batchnorm_modules(classifier_network)
    elif config.network_params['normlayer']  == 'identity':
        print('BN to identity')
        classifier_network = module_modification.convert_batchnorm_modules(classifier_network,
                                                                           module_modification.nullify_batchnorm_modules)
    elif config.network_params['normlayer'] == 'instance':
        print('BN to instance norm')
        classifier_network = module_modification.convert_batchnorm_modules(classifier_network,
                                                                           module_modification._batchnorm_to_instancenorm)

    classifier_network = classifier_network.to(config.DEVICE)
    # load_model = 'weights_best_train.pth'
    load_model = 'weights_best.pth'
    print('Loading : ', cparser.previous_model+ '/' + load_model)
    model_params_load(cparser.previous_model + '/' + load_model,
                      classifier_network, None, config.DEVICE)

    ### generate and save predictions + performance of previous model
    for pd_data in [pd_train, pd_test]:
        eval_dataloader = get_dataloaders_image(pd_data, file_tag=file_tag, utility_tag='utility',
                                                augmentations=test_transform, shuffle=False,
                                                num_workers=8, batch_size=config.BATCH_SIZE)

        utility_pred_l, utility_gt_l, group_gt_l = fast_epoch_evaluation(eval_dataloader, classifier_network,
                                                                         config.n_utility, config.DEVICE,
                                                                         groups=False)

        utility_pred_l = np.array(utility_pred_l).transpose()
        utility_decision = np.argmax(utility_pred_l, axis=-1)
        acc = (utility_decision == np.array(utility_gt_l))

        if cparser.loss == 'CE': #we always use CE
            loss = -1 * np.log(utility_pred_l[np.arange(utility_pred_l.shape[0]), utility_gt_l])
        else:
            print('Implement loss : ', cparser.type_loss)
            break

        pd_data['prevmodel_loss'] = loss
        pd_data['prevmodel_acc'] = acc
        pd_data['prevmodel_confidence'] = np.max(utility_pred_l, axis=-1)
        pd_data['prevmodel_outgt'] = utility_pred_l[np.arange(utility_pred_l.shape[0]), utility_gt_l]


    del classifier_network

    ################## EXPAND DATASET ##################


    ## load evaluation of previous model to identify the original training sample and the remaining (the ones that where not used during training, val or test)
    pd_eval = pd.read_csv(config.basedir + config.model_name + '/eval0.csv')
    pd_eval = pd_eval.sort_values(by=['sample_index']) #sample index is the sample unique id

    pd_all = pd.concat([pd_train, pd_test])
    pd_all = pd_all.sort_values(by=['sample_index'])

    check1 = np.sum(np.abs(pd_all['sample_index'].values - pd_eval['sample_index'].values))
    check2 = np.sum(np.abs(pd_all['utility'].values - pd_eval['ygt'].values))
    print('Sanity check (true,true) : ',check1 == 0,check2 == 0)
    if (check1 != 0) | (check2 != 0):
        print('Dataset compatibility error ---> ending')
        sys.exit()

    dataset_values = np.array(pd_eval['dataset'].values) #tags corresponding to train, val, test and other (other refers to samples that were excluded and are the ones we will incorporate)
    utility_values = np.array(pd_eval['ygt'].values)
    if 'group_tag' in pd_eval.columns:
        group_values = np.array(pd_eval['group_tag'].values)
    else:
        group_values = np.array(pd_eval['ygt'].values)

    if cparser.dataset == 'cifar10':
        utility_filter = np.array([1 if u in [cparser.addclass*2, cparser.addclass*2+1] else 0 for u in utility_values])
    if cparser.dataset == 'celebA_blond':
        utility_filter = np.array([1 if u in [cparser.addclass] else 0 for u in group_values])
    if cparser.dataset == 'CUB':
        utility_filter = np.array([1 if u in [cparser.addclass] else 0 for u in group_values])

    pd_all['dataset_previous'] = np.array(dataset_values)
    dataset_values[(dataset_values == 'other') & (utility_filter == 1)] = 'train' #samples that had the tag 'other'
    pd_all['dataset'] = dataset_values

    pd_train_split = pd_all.loc[pd_all['dataset'] =='train']
    pd_val_split = pd_all.loc[pd_all['dataset'] =='validation']
    pd_test = pd_all.loc[pd_all['dataset'] =='test']


    print('Train split : ')
    print(pd_train_split.groupby(group_tag_dataset)['dataset'].count())
    print(pd_train_split.groupby(group_tag_dataset)['dataset'].count()/len(pd_train_split))

    print('Val split : ')
    print(pd_val_split.groupby(group_tag_dataset)['dataset'].count())
    print(pd_val_split.groupby(group_tag_dataset)['dataset'].count()/len(pd_val_split))

    print('Test split : ')
    print(pd_test.groupby(group_tag_dataset)['dataset'].count())
    print(pd_test.groupby(group_tag_dataset)['dataset'].count()/len(pd_test))

    ######################################

    #### New Model
    nutility = len(pd_train['utility'].unique())

    network_params = {'network': cparser.network,
                      'normlayer': cparser.normlayer,
                      'pretrained': cparser.pretrained}

    optimizer_params = {'optimizer': cparser.optim,
                        'LEARNING_RATE': cparser.lr,
                        'weight_decay': cparser.optim_wreg}

    if cparser.patience == 0:  # patience for stopping criteria, zero means should train the selected epochs
        cparser.patience = cparser.epochs + 2

    ### Default parameters for all possible schedulers
    scheduler_params = {'steps_per_epoch': int(np.ceil(len(pd_train_split) / cparser.batch)),
                        'T_max': 200,
                        'max_lr': cparser.lr,
                        'milestones': [100, 150],
                        'gamma': 0.1,
                        'lr_decay': 0.9,
                        'patience': 5}  # patience for ManualLRDecayPlateau

    train_mode_params = {}

    ### Dataloaders ###
    sampler_on = False
    if cparser.train_mode in ['gmmf','grm']:  ## config for group modalities ('gmmf = group minmax fairness, 'gb': group balance)

        group_tag = group_tag_dataset
        group_prior = pd_train_split.groupby(group_tag)['utility'].count().values / len(pd_train_split)

        group_constrain = pd_all.loc[pd_all['dataset_previous'] == 'validation'].groupby(group_tag)['prevmodel_loss'].mean().values
        group_constrain_acc = pd_all.loc[pd_all['dataset_previous'] == 'validation'].groupby(group_tag)['prevmodel_acc'].mean().values

        if cparser.train_mode == 'gmmf':
            group_constrain = 0*group_constrain
            group_constrain_acc = 0*group_constrain_acc + 1
        #group_constrain_acc = pd_test.groupby(group_tag)['prevmodel_acc'].mean().values

        # print(pd_all.loc[pd_all['dataset_previous'] == 'val'].groupby(group_tag)['prevmodel_loss'].mean())

        # print(group_constrain)
        # print(group_constrain_acc)

        train_mode_params['group_constrain'] = group_constrain
        train_mode_params['group_constrain_acc'] = group_constrain_acc
        train_mode_params['group_prior'] = group_prior

        train_mode_params['max_weight_change'] = cparser.max_weight_change
        train_mode_params['cost_delta_improve'] = cparser.cost_delta_improve
        if cparser.min_weight_prior:
            train_mode_params['min_weight'] = cparser.min_weight * group_prior
        else:
            train_mode_params['min_weight'] = cparser.min_weight * np.ones_like(group_prior)
        train_mode_params['lr_penalty'] = cparser.lr_weight

        # print(train_mode_params['min_weight'], np.sum(train_mode_params['min_weight']))
        if np.sum(train_mode_params['min_weight']) > 1:
            train_mode_params['min_weight'] = train_mode_params['min_weight'] / np.sum(train_mode_params['min_weight'])

        weight_init = to_np(euclidean_proj_simplex(torch.from_numpy(group_prior), s=1, e=train_mode_params['min_weight']))
        train_mode_params['weights_init'] = weight_init



        print('group_prior :', train_mode_params['group_prior'])
        print('group_constrain :', train_mode_params['group_constrain'])
        print('group_constrain_acc :', train_mode_params['group_constrain_acc'])

        print('min weight :', train_mode_params['min_weight'])
        print('weight_init :', train_mode_params['weights_init'])
        group2cat = True

    else:
        group_tag = None
        group2cat = False

    weights_tag = None
    if cparser.train_mode == 'srm':
        weights_tag = 'weights'
        group_tag = 'sample_ix'

        pd_train_split[weights_tag] = 1
        pd_val_split[weights_tag] = 1

        pd_train_split[group_tag] = np.arange(len(pd_train_split)) #sample relative to the train set
        pd_val_split[group_tag] = np.arange(len(pd_val_split)) #sample relative to the val set

        train_mode_params['max_weight_change'] = cparser.max_weight_change
        train_mode_params['cost_delta_improve'] = cparser.cost_delta_improve
        train_mode_params['min_weight'] = cparser.min_weight
        train_mode_params['lr_penalty'] = cparser.lr_weight


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


    train_dataloader = get_dataloaders_image(pd_train_split, file_tag=file_tag, utility_tag='utility',
                                             group_tag=group_tag, weights_tag=weights_tag,
                                             augmentations=train_transform, shuffle=True,
                                             num_workers=8, batch_size=config.BATCH_SIZE, group2cat=group2cat,
                                             sampler_on=sampler_on)

    val_dataloader = get_dataloaders_image(pd_val_split, file_tag=file_tag, utility_tag='utility',
                                           group_tag=group_tag, weights_tag=weights_tag,
                                           augmentations=test_transform, shuffle=False,
                                           num_workers=8, batch_size=config.BATCH_SIZE, group2cat=group2cat)


    ## Include previous prediction per sample on a dataset attribute (needed only for SRM)
    if cparser.train_mode == 'srm':
        ###
        train_dataloader.dataset.previous_loss = pd_train_split['prevmodel_loss'].values
        val_dataloader.dataset.previous_loss = pd_val_split['prevmodel_loss'].values

        train_dataloader.dataset.previous_acc = pd_train_split['prevmodel_acc'].values
        val_dataloader.dataset.previous_acc = pd_val_split['prevmodel_acc'].values


    eval_dataloader = get_dataloaders_image(pd_all, file_tag=file_tag, utility_tag='utility',
                                            group_tag=None if group_tag is None else group_tag_dataset, #here we use the group_dataset
                                            augmentations=test_transform, shuffle=False,
                                            num_workers=8, batch_size=config.BATCH_SIZE, group2cat=group2cat)

    #### Model ####

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
        evaluation_list[i]['prevmodel_loss'] = pd_all['prevmodel_loss'].values
        evaluation_list[i]['prevmodel_acc'] = pd_all['prevmodel_acc'].values
        evaluation_list[i]['prevmodel_confidence'] = pd_all['prevmodel_confidence'].values
        evaluation_list[i]['prevmodel_outgt'] = pd_all['prevmodel_outgt'].values

        evaluation_list[i]['group_tag'] = pd_all[group_tag_dataset].values

        evaluation_list[i].to_csv(trainer.config.basedir + trainer.config.model_name + '/eval' + str(i) + '.csv',
                                  index=0)
        print('Saving : ', trainer.config.basedir + trainer.config.model_name + '/eval' + str(i) + '.csv')

