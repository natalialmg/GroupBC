import sys
sys.path.append("../")
import argparse
from general.utils import save_json,model_params_load,model_params_save
from general.misc import mkdir
from models.training import ManualLRDecayPlateau,ManualLRDecayNWReset,vanilla_trainer,epoch_vanilla_training,fast_epoch_evaluation
from models.robust_training import grm_trainer,groupDRO_trainer
import torch
import numpy as np
import pandas as pd
from torch import optim
from general.losses import losses,metrics

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self,argv, config_dict=None ):
        if config_dict is None:
            # args = self.parser.parse_args(argv)
            args, unknown = self.parser.parse_known_args(argv)
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])


class Config(argparse.Namespace):

    def __init__(self,n_utility=2, config_dic=None, **kwargs):

        self.basedir = 'models/'
        self.model_name = 'vanilla_model'
        self.best_model = 'weights_best.pth'
        self.best_model_train = 'weights_best_train.pth'
        self.last_model = 'weights_last.pth'
        self.seed = 42
        self.GPU_ID = 0


        self.BATCH_SIZE = 32
        self.n_workers = 32
        self.n_utility = n_utility


        self.network_params = {'network':'resnet34',
                               'normlayer':'batchnorm',
                               'pretrained':True}
        # self.batchnorm = True
        self.regression = False
        self.type_loss = 'CE'
        self.type_metric = []


        # Loss  -> todo!: I have to add regularizations


        self.EPOCHS = 100
        self.epochs_warmup = 100

        ##Optimizer
        # self.optimizer = 'adam'
        self.optimizer_params = {'optimizer':'adam',
                                 'LEARNING_RATE':1e-4,
                                 'weight_decay': 0}
        # self.optim_weight_decay = 0
        # self.LEARNING_RATE = 1e-4

        self.scheduler = None
        self.scheduler_params = {'T_max': 200}
        self.patience = 100
        self.n_print = 1
        self.eval_every_epochs = -1

        self.num_max_batch = np.infty #number of batches to consider in each epoch, if set to np.infty all batches will be considered
        self.train_mode = 'vanilla'
        self.train_mode_params = {}



        #### TO REVIEW ::: ######
        # self.lrdecay = 0.5
        # self.delta_epoch_decay = 50
        # self.penalty = 0
        # self.lr_weights = 0.1


        if (not self.regression) :
            self.type_metric = ['acc']
            # self.type_metric = ['err']

        if config_dic is not None:
            for key in config_dic.keys():
                setattr(self, key, config_dic[key])

        for k in kwargs:
            setattr(self, k, kwargs[k])

        torch.manual_seed(self.seed)
        if torch.cuda.is_available() and self.GPU_ID >= 0:
            DEVICE = torch.device('cuda:%d' % (self.GPU_ID))
        else:
            DEVICE = torch.device('cpu')
        self.DEVICE = DEVICE

    # def is_valid(self, return_invalid=False):
        # Todo! I have to update this properly
        # ok = {}
        #
        # if return_invalid:
        #     return all(ok.values()), tuple(k for (k, v) in ok.items() if not v)
        # else:
        #     return all(ok.values())
    def update_parameters(self, allow_new=True, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])
    def save_json(self,save_path=None):

        mkdir(self.basedir + self.model_name)

        config_dict = self.__dict__
        config2json = {}
        for key in config_dict.keys():
            if key != 'DEVICE':
                if type(config_dict[key]) is np.ndarray:
                    config2json[key] = config_dict[key].tolist()
                else:

                    if key == 'train_mode_params':
                        config2json[key] = {}
                        for key2 in config_dict[key].keys():
                            if type(config_dict[key][key2]) is np.ndarray:
                                config2json[key][key2] = config_dict[key][key2].tolist()
                    else:
                        config2json[key] = config_dict[key]
        if save_path is None:
            save_path =  self.basedir + self.model_name + '/config.json'
        save_json(config2json, save_path)
        print('Saving config json file in : ', save_path)



class Model():
    def __init__(self,config, train_dataloader, val_dataloader, classifier_network):

        #Load datasets; classifier and config file
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.classifier_network = classifier_network
        self.classifier_network = self.classifier_network.to(self.config.DEVICE)
        self.history = []

        ## Loss type ##
        if self.config.type_loss not in ['L1','L2','CE']:
            print('WARNING!! Loss type : ', self.config.type_loss, ', not specified correctly, will be set to L2 ')
            self.config.type_loss = 'L2'

        if self.config.regression & (self.config.type_loss == 'CE'):
            print('WARNING!!: CE LOSS WITH REGRESSION OBJECTIVE!')

        self.criterion = losses(type_loss=self.config.type_loss,reduction='sum',regression=self.config.regression)

        ## metric ##
        self.metric_dic = None
        if len(self.config.type_metric) > 0:
            self.metric_dic = {}
            for metric in self.config.type_metric:
                print(metric)
                self.metric_dic[metric] = metrics(type_loss = metric)


        ## Optimizer ##
        if self.config.optimizer_params['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.classifier_network.parameters(), lr=self.config.optimizer_params['LEARNING_RATE'],
                                        weight_decay=self.config.optimizer_params['weight_decay'])

        elif self.config.optimizer_params['optimizer'] == 'rms':
            self.optimizer = optim.RMSprop(self.classifier_network.parameters(), lr=self.config.optimizer_params['LEARNING_RATE'],
                                           weight_decay=self.config.optimizer_params['weight_decay'])

        else:
            self.optimizer = optim.SGD(self.classifier_network.parameters(), lr=self.config.optimizer_params['LEARNING_RATE'],
                                       momentum=0.9, weight_decay=self.config.optimizer_params['weight_decay'])

        ## Evaluation  ## ##todo!: I have to work a bit more on this
        self.df_train_result = []
        self.df_val_result = []
        self.scheduler = None

        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('---------------- Vanilla Trainer was CREATED ----------------------------')
        mkdir(self.config.basedir)
        mkdir(self.config.basedir+self.config.model_name)
        print('model directory:', self.config.basedir+self.config.model_name+'/')
        print()

        print('Config :')
        print(self.config)
        print()

        print('Network :')
        print(self.classifier_network)
        print()

        print('Optimizer :')
        print(self.optimizer)
        print('-------------------------------------------------------------------------')
        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    def train_model(self,epochs = 0,metric_stopper = 'loss',epoch_warmup=30,eval_dataloader = None,
                    epoch_training=epoch_vanilla_training, train_modality = 'vanilla'):

        if epochs>0:
            self.config.EPOCHS = epochs

        self.scheduler = None
        if self.config.scheduler == 'MultiStepLR':
            print('scheduler Multi steps (milestones, gamma)',
                  self.config.scheduler_params['milestones'],
                  self.config.scheduler_params['gamma'])
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=self.config.scheduler_params['milestones'],
                                                                  gamma=self.config.scheduler_params['gamma'])
        elif self.config.scheduler == 'OneCycleLR':
            steps_per_epoch = self.config.scheduler_params['steps_per_epoch']
            print('scheduler, steps_per_epoch :', steps_per_epoch,
                  ' max_lr : ', self.config.scheduler_params['max_lr'])
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR( self.optimizer,
                                                                  self.config.scheduler_params['max_lr'],
                                                                  epochs=epochs,
                                                                  steps_per_epoch=steps_per_epoch)
        elif self.config.scheduler == 'CosineAnnealingLR':
            self.config.scheduler_params['T_max'] = self.config.EPOCHS
            print('scheduler, T_max :', self.config.scheduler_params['T_max'] )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.scheduler_params['T_max'])
        elif self.config.scheduler == 'ManualLRDecayPlateau':
            if 'lr_decay' not in self.config.scheduler_params.keys():
                self.config.scheduler_params['lr_decay'] = 0.95
            if 'patience' not in self.config.scheduler_params.keys():
                self.config.scheduler_params['patience'] = 5
            print('scheduler ManulaLRDecayPlateau, (lr_decay, patience) :', self.config.scheduler_params['lr_decay'],
                  self.config.scheduler_params['patience'])

            self.scheduler = ManualLRDecayPlateau(self.config.scheduler_params['lr_decay'],
                                                  self.config.scheduler_params['patience'],
                                                  self.optimizer)

        elif self.config.scheduler == 'ManualLRDecayNWReset' :
            if 'lr_decay' not in self.config.scheduler_params.keys():
                self.config.scheduler_params['lr_decay'] = 0.95
            if 'patience' not in self.config.scheduler_params.keys():
                self.config.scheduler_params['patience'] = 5
            print('scheduler ManualLRDecayNWReset, (lr_decay, patience) :', self.config.scheduler_params['lr_decay'],
                  self.config.scheduler_params['patience'])

            model_params_save(self.config.basedir + self.config.model_name + '/weights_init.pth',
                              self.classifier_network,
                              self.optimizer)
            def model_reset_func(network,optimizer):
                model_params_load(self.config.basedir + self.config.model_name + '/weights_init.pth',
                                  network,
                                  optimizer,
                                  self.config.DEVICE)
            self.scheduler = ManualLRDecayNWReset(self.config.scheduler_params['lr_decay'],
                                                  self.config.scheduler_params['patience'],
                                                  self.optimizer, self.classifier_network,
                                                  model_reset_func)

        print('scheduler :: ', self.scheduler )

        ## DP-SGD
        # if self.config.dp :
        #     inspector = DPModelInspector()
        #     if inspector.validate(self.classifier_network):
        #
        #         privacy_engine = PrivacyEngine(self.classifier_network,
        #             sample_rate=self.config.sample_rate,
        #             epochs=self.config.EPOCHS,
        #             target_epsilon=self.config.epsilon,
        #             target_delta=self.config.delta,
        #             max_grad_norm=self.config.max_grad_norm)
        #
        #         privacy_engine.attach(self.optimizer)
        #         print("Using sigma=",privacy_engine.noise_multiplier, ' and C=' ,self.config.max_grad_norm)
        #     else:
        #         print('DP not possible due to model incompatibility')
        #         return

        if train_modality == 'vanilla':
            output = vanilla_trainer(self.train_dataloader, self.val_dataloader,
                           self.optimizer, self.classifier_network, self.criterion, self.config,
                            metrics_dic = self.metric_dic, eval_dataloader=eval_dataloader,
                                               metric_stopper = metric_stopper, reg_dic=None, reg_weights=None,
                            epoch_training = epoch_training,epoch_warmup=epoch_warmup, scheduler = self.scheduler)

        elif train_modality in ['group_minmax', 'group_constrain', 'group_minmax_erm', 'group_balance']:
            print('Group trainer')
            output = grm_trainer(self.train_dataloader, self.val_dataloader,
                                     self.optimizer, self.classifier_network, self.criterion, self.config,
                                     metrics_dic=self.metric_dic,
                                     eval_dataloader=eval_dataloader,
                                     metric_stopper=metric_stopper, reg_dic=None, reg_weights=None,
                                     epoch_warmup=epoch_warmup, scheduler = self.scheduler)

        elif train_modality in ['group_dro']:
            print('Group trainer')
            output = groupDRO_trainer(self.train_dataloader, self.val_dataloader,
                                     self.optimizer, self.classifier_network, self.criterion, self.config,
                                     metrics_dic=self.metric_dic,
                                     eval_dataloader=eval_dataloader,
                                     metric_stopper=metric_stopper, reg_dic=None, reg_weights=None,
                                     epoch_warmup=epoch_warmup, scheduler = self.scheduler)

        #
        # elif train_modality == 'group_dro':
        #     output = groupDRO_trainer(self.train_dataloader, self.val_dataloader,
        #                              self.optimizer, self.classifier_network, self.criterion, self.config,
        #                              metrics_dic=self.metric_dic, val_stopper=val_stopper,
        #                              eval_dataloader=eval_dataloader,
        #                              metric_stopper=metric_stopper, reg_dic=None, reg_weights=None,
        #                              epoch_warmup=epoch_warmup, scheduler = self.scheduler)
        #
        # elif train_modality == 'sample_constrain':
        #     output = penalty_trainer(self.train_dataloader, self.val_dataloader,
        #                     self.optimizer, self.classifier_network, self.criterion, self.config,
        #                     metrics_dic=self.metric_dic, val_stopper=val_stopper,
        #                     eval_dataloader=eval_dataloader,
        #                     metric_stopper=metric_stopper, reg_dic=None, reg_weights=None,
        #                     epoch_warmup=epoch_warmup, scheduler = self.scheduler)


        if eval_dataloader is None:
            self.history = output
        else:
            self.history = output[0]
            self.evaluation_list = output[1]

        return output



    def fast_epoch_evaluation_bundle(self, dataloader):

        columns_tag = ['group_gt', 'utility_gt']
        for _ in range(self.config.n_utility):
            columns_tag.append('utility_pest_'+str(_))

        ### train ###
        utility_pred_l, utility_gt_l, group_gt_l= fast_epoch_evaluation(dataloader, self.classifier_network,
                                                                         self.config.n_utility,self.config.DEVICE,
                                                                         groups = False)
        data_results = np.concatenate([np.array(group_gt_l)[:, np.newaxis],
                                             np.array(utility_gt_l)[:, np.newaxis],
                                             np.array(utility_pred_l).transpose()], axis=1)
        df_result = pd.DataFrame(data_results, columns=columns_tag)


        return df_result