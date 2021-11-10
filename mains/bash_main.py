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

train_mode = 'erm'
train_mode = 'gmmf'

dataset_reduction = 0.4

normlayer = 'batchnorm'
if dataset in ['cifar10']:
    pretrained = False
    net = 'resnet18'
    optim_wreg = 1e-4

if dataset in ['celebA_blond','CUB']:
    pretrained = True
    net = 'resnet34'
    optim_wreg = 1e-4

## Optimizer
optim = 'sgd'

train_mode_names = {'erm':'erm',
                    'gmmf':'gmmf'}


model_name_prefix = train_mode_names[train_mode] + '_' + net
if pretrained:
    model_name_prefix = model_name_prefix + '_pretrained'
model_name_prefix = model_name_prefix + '_' + normlayer


## Optimizer
if optim == 'sgd':
    if dataset in ['cifar10']:
        lr = 1e-1
        model_name_prefix = model_name_prefix + '_sgd1e1_'
    elif dataset in ['celebA_blond', 'CUB']:

        if train_mode in ['gmmf']:
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
elif dataset_reduction == 0.4:
    model_name_prefix = model_name_prefix + 'datared04_'
elif dataset_reduction == 0.1:
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
    epochs = 52

loss_list = ['CE']

## seed list
seed_list=[42]
split_list = [1]
gpu = 0


file_bash_name = dataset+'_bash.sh'


#(valid for gmmf)
group_param_dic = {'mw0wc05c025':[0.0,0.5,0.25]} #min_weight, max_weight_change, cost_delta_improve

# group_param_dic = {'pg08mw05cd025': [0.8, 0.5, 0.25]}

# min_weight_prior = True


with open(file_bash_name,'w') as f:
    for split in split_list:
        for seed in seed_list:
            for loss in loss_list:
                for gparam_key in group_param_dic.keys():
                    values = group_param_dic[gparam_key]

                    min_weight = values[0]
                    max_weight_change = values[1]
                    cost_delta_improve = values[2]

                    if train_mode in ['gmmf']:
                        model_name = model_name_prefix + gparam_key + '_'+ loss + '_seed' + str(seed) + '_split' + str(split)
                    else:
                        model_name = model_name_prefix + loss + '_seed' + str(seed) + '_split' + str(split)
                    out_file_ext = dataset + '_' + model_name + '_verbose'

                    cmd = 'python main.py --basedir="{}" --dataset="{}" --model_name="{}" --gpu={} --seed={} --split={} --augmentation={}'.format(basedir, dataset,model_name,
                                                                                                                                                  gpu, seed, split,augmentation)
                    cmd = cmd + ' --batch={} --network="{}" --pretrained={} --optim_wreg={} --optim="{}" --lr={}'.format(batchsize, net, pretrained,
                                                                                                                         optim_wreg, optim, lr)
                    cmd = cmd + ' --loss="{}" --epochs={} --min_weight={} --max_weight_change={} --cost_delta_improve={}'.format(loss, epochs, min_weight,
                                                                                                                                 max_weight_change, cost_delta_improve)
                    cmd = cmd + ' --normlayer="{}" --dataset_reduction={} --train_mode="{}" --scheduler="{}" > {}.txt '.format(normlayer, dataset_reduction,
                                                                                                                               train_mode, scheduler, out_file_ext)

                    f.write(cmd+'\n\n\n')
                    f.write('\n\n\n')