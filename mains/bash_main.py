# dataset ='celebA_blond'
dataset ='CUB'
# dataset = 'cifar10'



basedir = '/data/natalia/models/' + dataset + '/'


## Network
loadflag= False
batchnorm = True
augmentation = True

scheduler = None
# scheduler = 'CosineAnnealingLR'
# scheduler = 'ManualLRDecayPlateau'
scheduler = 'ManualLRDecayNWReset'
# scheduler = 'OneCycleLR'
# scheduler = 'MultiStepLR'

# train_mode = 'vanilla'
# train_mode = 'group_dro'
train_mode = 'group_minmax'

# train_mode = 'group_minmax_erm'
# train_mode = 'group_minmax_balance'


dataset_reduction = 1
normlayer = 'batchnorm'
if dataset in ['cifar10']:
    pretrained = False
    net = 'resnet18'
    optim_wreg = 1e-4

if dataset in ['cifar100_coarse','cifar100_coarse_landanimal']:
    pretrained = False
    net = 'resnet18'
    optim_wreg = 1e-4

if dataset in ['celebA_blond','CUB']:
    pretrained = True
    net = 'resnet34'
    optim_wreg = 1e-4

## Optimizer
optim = 'sgd'


train_mode_names = {'vanilla':'vanilla',
                    'group_minmax':'gminmax',
                    'group_dro': 'gdro',
                     'group_minmax_erm': 'gminmaxerm',
                    'group_minmax_balance': 'gminmaxbal'}





model_name_prefix = train_mode_names[train_mode] + '_' + net
if pretrained:
    model_name_prefix = model_name_prefix + '_pretrained'
model_name_prefix = model_name_prefix + '_' + normlayer


## Optimizer
if optim == 'sgd':
    if dataset in ['cifar10', 'cifar100_coarse','cifar100_coarse_landanimal']:
        lr = 1e-1
        model_name_prefix = model_name_prefix + '_sgd1e1_'
    elif dataset in ['celebA_blond', 'CUB']:

        if train_mode in ['group_minmax','group_dro']:
            lr = 1e-4
            model_name_prefix = model_name_prefix + '_sgd1e4_'
        else:
            lr = 1e-3
            model_name_prefix = model_name_prefix + '_sgd1e3_'

elif optim == 'adam':
    lr = 1e-4
    model_name_prefix = model_name_prefix + '_adam1e4_'


## Scheduler
if scheduler is not None:
    model_name_prefix = model_name_prefix +  scheduler + '_'

if not augmentation:
    model_name_prefix = model_name_prefix + 'noaug_'

#optimizer l2 regularization
if optim_wreg > 0:
    if optim_wreg == 1e-3:
        model_name_prefix = model_name_prefix + 'reg1e3_'
    elif optim_wreg == 1e-4:
        model_name_prefix = model_name_prefix + 'reg1e4_'
    elif optim_wreg == 1e-2:
        model_name_prefix = model_name_prefix + 'reg1e2_'
    elif optim_wreg == 5e-3:
        model_name_prefix = model_name_prefix + 'reg5e3_'
    elif optim_wreg == 5e-4:
        model_name_prefix = model_name_prefix + 'reg5e4_'
    elif optim_wreg == 1e-1:
        model_name_prefix = model_name_prefix + 'reg1e1_'
    elif optim_wreg == 1:
        model_name_prefix = model_name_prefix + 'reg1_'


## dataset reduction?
if dataset_reduction == 0.6:
    model_name_prefix = model_name_prefix + 'datared06_'
if dataset_reduction == 0.4:
    model_name_prefix = model_name_prefix + 'datared04_'
if dataset_reduction == 0.1:
    model_name_prefix = model_name_prefix + 'datared01_'

## dataloaders
if 'cifar' in dataset:
    batchsize=128 #128 is default
    if 'cifar10' in dataset:
        epochs = 152
    else:
        epochs = 152

elif dataset == 'celebA_blond':
    batchsize = 64 # 128 is default
    epochs = 52

elif dataset == 'CUB':
    batchsize = 64  # 128 is default
    epochs = 102


loss_list = ['CE']

## seed list
seed_list=[42,43,44,45,46]
seed_list=[42]
seed_list=[42,43,44,45,46]
seed_list=[44,45,46]

# seed_list=[46]
split_list = [1]
gpu = 0

# if train_mode in ['group_minmax']:
# group_param_dic = {'e0mw05cd025':[0,0.5,0.25]}
group_param_dic = {'e0mw025cd025':[0,0.25,0.25]}
group_param_dic = {'e0mw01cd025':[0,0.1,0.25]}
# group_param_dic = {'e0mw05cd025':[0,0.5,0.25]}

group_param_dic = {'e001mw05cd025':[0.01,0.5,0.25]}
group_param_dic = {'e0001mw05cd025':[0.001,0.5,0.25]}
# group_param_dic = {'e01mw05cd025':[0.1,0.5,0.25]}
# group_param_dic = {'e025mw05cd025':[0.25,0.5,0.25]}

file_bash_name = dataset+'_bash.sh'


# group_param_dic = {'pg05mw05cd025':[0.5,0.5,0.25]}


# group_param_dic = {'pg0mw05cd025':[0.0,0.5,0.25]}

group_param_dic = {'pg0mw05cd025':[0.0,0.5,0.25],
                   'pg0001mw05cd025': [0.001, 0.5, 0.25],
                   'pg001mw05cd025': [0.01, 0.5, 0.25],
                   'pg01mw05cd025': [0.1, 0.5, 0.25],
                   'pg05mw05cd025': [0.5, 0.5, 0.25],
                   'pg08mw05cd025': [0.8, 0.5, 0.25],
                   'pg1mw05cd025': [1.0, 0.5, 0.25]}



# group_param_dic = {'pg1mw05cd025': [1.0, 0.5, 0.25]}


group_param_dic = {'pg0mw05cd025':[0.0,0.5,0.25],
                   'pg0001mw05cd025': [0.001, 0.5, 0.25],
                   'pg001mw05cd025': [0.01, 0.5, 0.25],
                   'pg01mw05cd025': [0.1, 0.5, 0.25],
                   'pg05mw05cd025': [0.5, 0.5, 0.25],
                   'pg08mw05cd025': [0.8, 0.5, 0.25]}

# group_param_dic = {'pg07mw05cd025': [0.7, 0.5, 0.25],
                   # 'pg09mw05cd025': [0.9, 0.5, 0.25]}

group_param_dic = {'pg1mw05cd025': [1.0, 0.5, 0.25]}

if train_mode not in ['group_minmax']:
    group_param_dic = {'':[0.0,0.5,0.25]}

# group_param_dic = {'pg08mw05cd025': [0.8, 0.5, 0.25]}


min_weight_prior = True

with open(file_bash_name,'w') as f:
    for split in split_list:
        for seed in seed_list:
            for loss in loss_list:

                for gparam_key in group_param_dic.keys():
                    values = group_param_dic[gparam_key]
                    min_weight = values[0]
                    max_weight_change = values[1]
                    cost_delta_improve = values[2]
                    if train_mode in ['group_minmax']:
                        model_name = model_name_prefix + gparam_key + '_'+ loss + '_seed' + str(seed) + '_split' + str(split)
                    else:
                        model_name = model_name_prefix + loss + '_seed' + str(seed) + '_split' + str(split)
                    out_file_ext = dataset + '_' + model_name + '_verbose'

                    cmd = 'python main.py --basedir="{}" --dataset="{}" --model_name="{}" --gpu={} --seed={} --split={} --augmentation={}'.format(basedir, dataset,model_name,
                                                                                                                                                  gpu, seed, split,augmentation)
                    cmd = cmd + ' --batch={} --network="{}" --pretrained={} --optim_wreg={} --optim="{}" --lr={} --min_weight_prior={}'.format(batchsize, net, pretrained,
                                                                                                                         optim_wreg, optim, lr,min_weight_prior)
                    cmd = cmd + ' --loss="{}" --epochs={} --min_weight={} --max_weight_change={} --cost_delta_improve={}'.format(loss, epochs, min_weight,
                                                                                                                                 max_weight_change, cost_delta_improve)
                    cmd = cmd + ' --normlayer="{}" --dataset_reduction={} --train_mode="{}" --scheduler="{}" > {}.txt '.format(normlayer, dataset_reduction,
                                                                                                                               train_mode, scheduler, out_file_ext)

                    # run_command(cmd, minmem=1.5, use_env_variable=True, admissible_gpus=[0,1],sleep=40)
                    f.write(cmd+'\n\n\n')
                    f.write('\n\n\n')