
import sys
sys.path.append("../")
import torch.nn as nn
import numpy as np
import pandas as pd


from general.utils import TravellingMean,to_np,model_params_load,model_params_save


class ManualLRDecayPlateau(object):
    def __init__(self, lr_decay, patience, optimizer, best_loss = None):
        self.patience = patience
        self.lr_decay = lr_decay
        self.counter = 0
        self.optimizer = optimizer
        self.best_loss = best_loss

    def step(self,loss):
        self.counter = np.maximum(self.counter-1,0)
        if self.best_loss is None:
            self.best_loss = loss
            print('saving initial best loss ', self.best_loss)
        else:
            if (self.best_loss<=loss) & (self.counter == 0):
                print('LR decay.....')
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay  # apply lrdecay
                self.counter = self.patience + 0
            elif (loss < self.best_loss) :
                self.best_loss = loss + 0
                print('update best loss to ', self.best_loss)


class ManualLRDecayNWReset(object):
    def __init__(self,
                 lr_decay, patience,
                 optimizer, network,
                 model_reset_func,
                 best_loss = None,best_loss_nw = None):

        self.patience = patience
        self.lr_decay = lr_decay
        self.counter = 0
        self.optimizer = optimizer
        self.network = network
        self.model_reset_func = model_reset_func

        self.best_loss = best_loss
        self.best_loss_nw = best_loss_nw
        self.counter_nw_reset = 0

    def step(self,loss,loss_nw):
        self.counter = np.maximum(self.counter-1,0)
        if self.best_loss is None:
            self.best_loss = loss
            print('saving initial best loss ', self.best_loss)
        else:
            if (self.best_loss<=loss) & (self.counter == 0):
                print('LR decay.....')
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay  # apply lrdecay
                self.counter = self.patience + 0
            elif (loss < self.best_loss) :
                self.best_loss = loss + 0
                print('update best loss to ', self.best_loss)

        if self.best_loss_nw is None:
            self.best_loss_nw = loss_nw
            print('saving initial best loss nw reset ', self.best_loss_nw)
        else:
            if (self.best_loss_nw <= loss_nw): #no improvement i
                print('NO IMPROVEMENT')
                if (self.counter_nw_reset < self.patience):
                    self.counter_nw_reset += 1
                else:
                    print('reset network ')
                    self.model_reset_func(self.network, self.optimizer)
                    self.best_loss_nw = None
                    self.best_loss = None
                    self.counter = 0
                    self.counter_nw_reset = 0

            else:
                self.counter_nw_reset = 0
                self.best_loss_nw = loss_nw
                print('update best loss nw to ', self.best_loss_nw)


class early_stopping(object):
    def __init__(self, patience, counter, best_loss):
        self.patience = patience #max number of nonimprovements until stop
        self.counter = counter #number of consecutive nonimprovements
        self.best_loss = best_loss

    def evaluate(self, loss):
        save = False #save nw
        stop = False #stop training
        if loss < self.best_loss:
            self.counter = 0
            self.best_loss = loss
            save = True
            stop = False
        else:
            self.counter += 1
            if self.counter > self.patience:
                stop = True

        return save, stop


def fast_epoch_evaluation(dataloader, classifier_network, n_utility,DEVICE, groups = False):

    # loss summary lists
    utility_pred_l = [[] for _ in range(n_utility)]
    utility_gt_l = []
    group_gt_l = []
    classifier_network = classifier_network.eval()

    # Loop through samples and evaluate
    for i_batch, sample_batch in enumerate(dataloader):
        if groups:
            x, utility, group = sample_batch
        else:
            x, utility = sample_batch
            group = utility

        x = x.to(DEVICE)

        # forward pass
        logits = classifier_network(x)
        softmax = nn.Softmax(dim=-1)(logits)

        ### SAVES FOR VISUALIZATION ###
        softmax_np = to_np(softmax)

        for _ in range(n_utility):
            utility_pred_l[_].extend(list(softmax_np[:,_]))

        if len(utility.shape) > 1:
            utility_gt_l.extend(list(to_np(utility).argmax(-1)))
        else:
            utility_gt_l.extend(list(to_np(utility)))
        if len(group.shape)>1:
            group_gt_l.extend(list(to_np(group).argmax(-1)))
        else:
            group_gt_l.extend(list(to_np(group)))

    return utility_pred_l, utility_gt_l, group_gt_l


def epoch_vanilla_training(dataloader, classifier_network, criterion,
                         optimizer, DEVICE, metrics_dic = None,
                           reg_dic=None, reg_weights=None,
                           train_type=True, scheduler = None,
                           num_max_batch = np.infty ):

    #dataloader
    #classifier_network
    #criterion: loss function
    #reg_dic: dictionary with {name of regularization: regularization function}
    #reg_weights: dictionary with {name of regularization: weight}

    if train_type:
        classifier_network.train()
    else:
        classifier_network.eval()

    output = {}
    output['criterion'] = TravellingMean()
    output['loss'] = TravellingMean()
    if metrics_dic is not None:
        for key in metrics_dic.keys():
            output[key] = TravellingMean()

    if reg_dic is not None:
        for key in reg_dic.keys():
            output[key] = TravellingMean()

    for i_batch, sample_batch in enumerate(dataloader):
        x, utility = sample_batch
        x = x.to(DEVICE)
        utility = utility.to(DEVICE)  # batch x nutility

        # zero the parameter gradients
        optimizer.zero_grad()

        # get output and loss
        logits = classifier_network(x)

        ######## Loss #########
        loss = criterion(logits, utility)

        # Mean
        loss = loss.mean()
        output['criterion'].update(np.array([to_np(loss)])) # update rolling loss

        if reg_dic is not None:  #if we have regularizations
            for tag in reg_dic.keys(): #compute each regularization
                reg = reg_dic[tag]
                loss_r = reg.forward(classifier_network) #we assume they are a function only of the network parameters

                if reg_weights is None:
                    loss += loss_r
                else:
                    loss += reg_weights[tag] * loss_r

                ## Update regularization
                output[tag].update(np.array([to_np(loss_r)]))

        # update rolling full loss (loss + reg)
        output['loss'].update(np.array([to_np(loss)]))

        if train_type:
            # backpropagation
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                # print('scheduler step')
                scheduler.step()

        #######  Metrics #######
        if metrics_dic is not None:
            for key in metrics_dic.keys():
                metric = metrics_dic[key](logits,utility)
                metric = metric.mean()
                output[key].update(np.array([to_np(metric)]))

        if num_max_batch <= i_batch:
            break


    for key in output.keys():
        output[key] = output[key].mean #save only mean

    return output


def vanilla_trainer(train_dataloader, val_dataloader, optimizer,
                    classifier_network, criterion, config, eval_dataloader=None,
                    metrics_dic=None, metric_stopper='loss',
                    reg_dic=None, reg_weights=None, epoch_training=epoch_vanilla_training,
                    epoch_warmup=30, scheduler = None):
    learning_rate_all = np.zeros([config.EPOCHS+1])
    history_tags = ['loss', 'criterion']

    history = {}
    history['loss_train'] = []
    history['loss_val'] = []
    history['criterion_train'] = []
    history['criterion_val'] = []

    if reg_dic is not None:
        for reg in reg_dic.keys():
            history[reg + '_train'] = []
            history[reg + '_val'] = []
            history_tags.append(reg)

    if metrics_dic is not None:
        for metric in metrics_dic.keys():
            history[metric + '_train'] = []
            history[metric + '_val'] = []
            history_tags.append(metric)

        if metric_stopper not in metrics_dic.keys():
            metric_stopper = 'loss'

    if eval_dataloader is not None:
        evaluation_list = []

    #stopper train and validation
    metric_stopper_train = metric_stopper + '_train'
    metric_stopper_val = metric_stopper + '_val'

    stop = False
    epoch = 0
    epoch_best = 0

    if config.scheduler == 'OneCycleLR':
        scheduler_batch = scheduler
        print('scheduler batch :', scheduler_batch)
    else:
        scheduler_batch = None

    while ((not stop) | (epoch < epoch_warmup)) & (epoch <= config.EPOCHS):

        # save learning rate
        learning_rate_all[epoch] = optimizer.param_groups[0]['lr'] + 0

        output_train = epoch_training(train_dataloader, classifier_network, criterion,
                                      optimizer, config.DEVICE, metrics_dic=metrics_dic,
                                      train_type=True, reg_dic=reg_dic, reg_weights=reg_weights,
                                      scheduler = scheduler_batch, num_max_batch = config.num_max_batch)

        output_val = epoch_training(val_dataloader, classifier_network, criterion,
                                    optimizer, config.DEVICE, metrics_dic=metrics_dic,
                                    train_type=False, reg_dic=reg_dic, reg_weights=reg_weights)

        ## history update
        history['loss_train'].append(output_train['loss'])
        history['loss_val'].append(output_val['loss'])
        history['criterion_train'].append(output_train['criterion'])
        history['criterion_val'].append(output_val['criterion'])

        if metrics_dic is not None:
            for metric in metrics_dic.keys():
                history[metric + '_train'].append(output_train[metric])
                history[metric + '_val'].append(output_val[metric])

        if reg_dic is not None:
            for reg in reg_dic.keys():
                history[reg + '_train'].append(output_train[reg])
                history[reg + '_val'].append(output_val[reg])

        # save last model
        model_params_save(config.basedir + config.model_name + '/' + config.last_model, classifier_network,
                          optimizer)

        ## loss for evaluation -> stopping criteria
        if ('auc' in metric_stopper) | ('acc' in metric_stopper) | ('softacc' in metric_stopper):
            loss_eval = 1 - history[metric_stopper_val][-1] + 0
            loss_eval_train = 1 - history[metric_stopper_train][-1] + 0
        else:
            loss_eval = history[metric_stopper_val][-1] + 0
            loss_eval_train = history[metric_stopper_train][-1] + 0

        if (epoch == 0):
            # init stopper validation
            stopper = early_stopping(config.patience, 0, loss_eval)
            model_params_save(config.basedir + config.model_name + '/' + config.best_model, classifier_network,
                              optimizer)
            epoch_best = epoch + 1

            stopper_train = early_stopping(config.patience, 0, loss_eval_train)
            model_params_save(config.basedir + config.model_name + '/'+ config.best_model_train, classifier_network,
                              optimizer)
            epoch_best_train = epoch + 1

        else:
            save, stop = stopper.evaluate(loss_eval)

            if save:
                epoch_best = epoch + 1
                print('saving best model, epoch: ', epoch_best)
                model_params_save(config.basedir + config.model_name + '/' + config.best_model, classifier_network,
                                  optimizer)

            #save best train
            save_train, stop_train = stopper_train.evaluate(loss_eval_train)
            if save_train:
                epoch_best_train = epoch + 1
                print('saving best train model, epoch: ', epoch_best_train)
                model_params_save(config.basedir + config.model_name + '/'+ config.best_model_train, classifier_network, optimizer)

        # print
        if ((epoch % config.n_print == config.n_print - 1) & (epoch >= 1)) | (epoch == 0):
            string_print = 'Epoch: ' + str(epoch) + '; lr: ' + str(np.round(optimizer.param_groups[0]['lr'],5))
            for tag in history_tags:
                string_print = string_print + '|' + tag + ' (tr,val): ' + str(
                    np.round(history[tag + '_train'][epoch], 3)) + \
                               ', ' + str(np.round(history[tag + '_val'][epoch], 3))

            string_print = string_print + '|stop_c : ' + str(stopper.counter) + '; best_train : ' +\
                           str(np.round(stopper_train.best_loss,3)) + ', epoch' + str(epoch_best_train)
            print(string_print + '; best_val : ' + str(np.round(stopper.best_loss,3)) + ', epoch' + str(epoch_best) )

        #OPTIONAL: EVALUATION AND SAVING EVERY  'config.eval_every_epochs' or in the last epoch using best training model
        save_eval_flag = (eval_dataloader is not None) &\
                         (((epoch % config.eval_every_epochs == 0) &\
                         (config.eval_every_epochs>0)) | stop | (epoch == config.EPOCHS))
        if save_eval_flag:

            print('**Saving Evaluation on epoch ', epoch)
            ## Evaluate:
            model_params_load(config.basedir + config.model_name + '/'+ config.best_model_train,
                              classifier_network,
                              None,
                              config.DEVICE)
            utility_pred_l, utility_gt_l, _ = fast_epoch_evaluation(eval_dataloader, classifier_network,
                                                                    config.n_utility, config.DEVICE,
                                                                    groups=False)
            utility_pred_l = np.array(utility_pred_l).transpose()
            columns = ['ypred_' + str(i) for i in range(utility_pred_l.shape[1])]
            pd_eval = pd.DataFrame(data=utility_pred_l, columns=columns)
            pd_eval['ygt'] = utility_gt_l
            pd_eval['epochs'] = epoch + 1
            evaluation_list.append(pd_eval)

            #resume last model:
            model_params_load(config.basedir + config.model_name + '/' + config.last_model,
                              classifier_network,
                              optimizer,
                              config.DEVICE)

        if (scheduler is not None) &  (config.scheduler in ['CosineAnnealingLR','MultiStepLR']):
            print('scheduler step')
            scheduler.step()

        if (scheduler is not None) &  (config.scheduler in ['ManualLRDecayPlateau']):
            print('scheduler step ')
            scheduler.step(history['loss_train'][-1])

        epoch += 1

    # -------- END TRAINING --------#

    # load best network
    print('Training Ended, loading best model : ', config.basedir + config.model_name + '/' + config.best_model)
    model_params_load(config.basedir + config.model_name + '/' + config.best_model, classifier_network, optimizer,
                      config.DEVICE)

    ## Eval best validation model ->
    if (eval_dataloader is not None):
        utility_pred_l, utility_gt_l, _ = fast_epoch_evaluation(eval_dataloader, classifier_network,
                                                                config.n_utility, config.DEVICE,
                                                                groups=False)
        utility_pred_l = np.array(utility_pred_l).transpose()
        columns = ['ypred_' + str(i) for i in range(utility_pred_l.shape[1])]
        pd_eval = pd.DataFrame(data=utility_pred_l, columns=columns)
        pd_eval['ygt'] = utility_gt_l
        pd_eval['epochs'] = epoch_best + 1
        evaluation_list.append(pd_eval)

        return history,evaluation_list
    else:
        return history
