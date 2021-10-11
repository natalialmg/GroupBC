import sys
sys.path.append("../")
import torch
from general.utils import TravellingMean,to_np,model_params_save,model_params_load
from models.training import fast_epoch_evaluation,early_stopping
import pandas as pd
import numpy as np

## Projections and PGA

def euclidean_proj_simplex(v, s=1, e=0):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem :
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= e
    We do this by applying the change of variables:
    h = w-e,
    s_aux = s - e \times v.shape[0],
    v_aux = v - e,
    And solving the euclidean on a positive simplex (using the algorithm from [1]):
    h^* = min_h 0.5 * || h - v_aux ||_2^2 , s.t. \sum_i h_i = s_aux, h_i >= 0

    Then our solution is w^* = h^* + e

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D

    # change variables
    v_aux = v - e
    s_aux = s - e * n

    #### Here we apply the original algorithm [1] using the auxiliar variables

    # check if we are already on the simplex
    if v_aux.sum() == s_aux and (v_aux >= 0).all():
        # best projection: itself!
        return v_aux
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.flip(torch.sort(v_aux)[0], dims=(0,))
    cssv = torch.cumsum(u, dim=0)
    # get the number of > 0 components of the optimal solution
    non_zero_vector = torch.nonzero(u * torch.arange(1, n + 1) > (cssv - s_aux), as_tuple=False)
    if len(non_zero_vector) == 0:
        rho = 0.0
    else:
        rho = non_zero_vector[-1].squeeze()
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s_aux) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    h = (v_aux - theta).clamp(min=0)

    #### redo variables
    w = h + e

    return w

def pga_weights(loss, weight, eta=0.1, iterations=1, decay=0.75, eta_increase_patience=5,
                   cost_delta_improve=0.05, max_weight_change=0.15, s=1, e=0):
    weight_list = []
    cost_list = []
    eta_list = []

    weight_list.append(np.array(weight))
    loss = np.array(loss)
    cost_list.append(np.mean(weight * loss) / np.mean(weight))

    it = 0
    p_increase = 0
    patience = 0
    # print(cost_list[-1])
    # print(weight)

    string_0 = 'eta_0 ' + str(eta)

    while (it < iterations):
        it += 1
        # weight_i = np.maximum(weight_list[-1] + eta * loss,0)
        weight_i = weight_list[-1] + eta * loss
        weight_i = euclidean_proj_simplex(torch.from_numpy(weight_i), s=s, e=e) # guarantee that it is in the simplex
        weight_i = to_np(weight_i)

        if np.min(weight_i) < 0:
            print('Warning: negative weight encountered')

        cost_i = np.mean(weight_i * loss) / np.mean(weight_i)
        relative_improvement = (cost_i - cost_list[0]) / cost_list[0]
        relative_weight_improvement = np.sum(np.abs(weight_i - weight_list[0])) / np.sum(np.abs(weight_list[0]))

        if (cost_i <= cost_list[-1]):
            patience += 1
        else:
            patience = 0

        if patience == 20:
            string_print = string_0 + '; eta_T ' + str(eta) + \
                           '; terminate: no weight change {:.3e}'.format(cost_i - cost_list[-1])
            break  # enough improvement

        if cost_i < cost_list[-1]:  # if step did not improve, decay eta and continue
            eta = eta * decay
            p_increase = 0
            string_print = string_0 + '; eta_T ' + str(eta) + \
                           '; terminate: max iterations '
            print('decrease', cost_i, cost_list[-1], weight_i, weight, weight_i>weight)
            continue

        if (relative_improvement > 1.2 * cost_delta_improve) | (relative_weight_improvement > 1.2 * max_weight_change):
            eta = eta * decay
            p_increase = 0
            string_print = string_0 + '; eta_T ' + str(eta) + \
                           '; terminate: max iterations '
            print('decrease because too much improvement')
            continue

        p_increase += 1
        if p_increase > eta_increase_patience:  # increase eta if we have seen improvements > patience
            # eta=eta/decay
            eta = eta * 2 / (decay + 1)
            p_increase = 0
            # print('increase')

        cost_list.append(cost_i)
        weight_list.append(weight_i)
        eta_list.append(eta)

        ####-stopping criteria-####

        # -enough improvement
        if relative_improvement > cost_delta_improve:
            # print('enough improvement ')
            string_print = string_0 + '; eta_T ' + str(eta) + \
                           '; terminate: enough cost improvement {:.3e}'.format(relative_improvement)
            break  # enough improvement

        if relative_weight_improvement > max_weight_change:
            # print('enough improvement ')
            string_print = string_0 + '; eta_T ' + str(eta) + \
                           '; terminate: enough weight improvement {:.3e}'.format(relative_weight_improvement)
            break  # enough improvement

        # -eta already too small
        if np.max(np.abs(eta * loss)) < 1e-20:
            # print('out low eta*loss ')
            string_print = string_0 + '; eta_T ' + str(eta) + \
                           '; terminate: low eta*loss {:.3e}'.format(np.max(np.abs(eta * loss)))
            break

        # -iterations
        string_print = string_0 + '; eta_T ' + str(eta) + \
                       '; terminate: max iterations '

    # if verbose:
    print(string_print)
    return weight_list, cost_list



######### GROUP BC ###########

def  grm_epoch_training(dataloader, classifier_network, criterions, weights,
                         optimizer, DEVICE, train_type= True, metrics_dic = None,
                         reg_dic=None, reg_weights=None,scheduler=None, num_max_batch = np.infty):

    '''
    This function train or evaluates an epoch TODO:!
    #inputs:
    dataloader, optimizer, classifier_network
    criterions: function that provides a base_loss
    train_type: if train performs backprop y otherwise only evaluates

    '''

    ###    INITIALIZE MEAN OBJECTS  #######

    #loss summary lists

    # print( dataloader.dataset[0][2])
    ngroups = weights.shape[0]
    weights = torch.from_numpy(weights).to(DEVICE)

    output = {}
    output['criterion'] = TravellingMean()
    output['criterion_group'] = [TravellingMean() for _ in range(ngroups)]
    output['loss'] = TravellingMean()

    if metrics_dic is not None: #additional metrics
        for key in metrics_dic.keys():
            output[key] = TravellingMean()
            output[key+'_group'] = [TravellingMean() for _ in range(ngroups)]

    if reg_dic is not None:
        for key in reg_dic.keys():
            output[key] = TravellingMean()

    if train_type:
        classifier_network = classifier_network.train()
    else:
        classifier_network = classifier_network.eval()

    for i_batch, sample_batch in enumerate(dataloader):

        x, utility, group = sample_batch
        x = x.to(DEVICE)
        utility = utility.to(DEVICE)  # batch x nutility
        group = group.to(DEVICE)  # batch x ngroups

        # zero the parameter gradients
        optimizer.zero_grad()

        # get output and losses
        logits = classifier_network(x)
        mass = utility.shape[0]  # number samples batch

        #loss
        base_loss = criterions(logits, utility)
        # print(base_loss.shape, utility.shape)
        output['criterion'].update(val=to_np(base_loss))  # base loss

        #group loss
        group_loss = base_loss.unsqueeze(-1) * group  # size batch x groups
        loss = torch.mean(torch.sum(group_loss * weights.unsqueeze(0), axis=1))

        group_loss = torch.sum(group_loss, axis=0)
        ns_groups = group.sum(0)  # number of samples per groups
        group_loss = group_loss / torch.max(ns_groups, torch.ones_like(ns_groups)) #size ngroups

        # loss = torch.mean(torch.sum(base_loss.unsqueeze(-1) * group * weights.unsqueeze(0), axis=1))  #DRO get maximum

        ### metrics
        if metrics_dic is not None:
            metric_eval_dic = {}
            for key in metrics_dic.keys():

                metric = to_np(metrics_dic[key](logits, utility))
                output[key].update(np.array([metric]))
                metric = np.sum(metric[..., np.newaxis] * to_np(group), axis=0)  # size batch x groups -> size groups
                metric_eval_dic[key] = metric / to_np(torch.max(ns_groups, torch.ones_like(ns_groups)))

        #save groups
        for g in range(ngroups):
            output['criterion_group'][g].update(val=np.array([to_np(group_loss[g])]),mass=ns_groups[g])
            if metrics_dic is not None:
                for key in metrics_dic.keys():
                    output[key+'_group'][g].update(val=np.array([metric_eval_dic[key][g]]),mass=ns_groups[g])

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

        output['loss'].update(val=to_np(loss), mass=mass)  # dro loss!

        if train_type :

            #classifier backpropagation
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                # print('scheduler step')
                scheduler.step()

        if num_max_batch <= i_batch:
            break

    ################ Final Epoch performance ######################################
    for key in output.keys(): #return only means
        if 'group' in key:
            for i in range(len(output[key])):
                output[key][i] = output[key][i].mean
        else:
            output[key] = output[key].mean

    return output


def grm_trainer(train_dataloader, val_dataloader, optimizer,
                    classifier_network, criterion, config, eval_dataloader=None,
                    metrics_dic=None, metric_stopper='loss',
                    reg_dic=None, reg_weights=None, epoch_training=grm_epoch_training,
                    epoch_warmup=30,scheduler=None):


    # config.train_mode_params['group_constrain']
    # config.train_mode_params['weights_init']
    # config.train_mode_params['group_prior']
    # config.train_mode_params['lr_penalty']
    # config.train_mode_params['min_weight']
    # config.train_mode_params['cost_delta_improve'],
    # config.train_mode_params['max_weight_change'],

    group_prior = config.train_mode_params['group_prior']

    learning_rate_all = np.zeros([config.EPOCHS+1])
    ngroups = group_prior.shape[0]

    history_tags = ['loss',
                    'criterion',
                    'criterion_group',
                    'constrain']

    history = {}
    history['loss_train'] = []
    history['loss_val'] = []
    history['criterion_train'] = []
    history['criterion_val'] = []
    history['constrain_train'] = []
    history['constrain_val'] = []

    history['criterion_group_train'] = []
    history['criterion_group_val'] = []
    history['weights_group'] = []

    if reg_dic is not None:
        for reg in reg_dic.keys():
            history[reg + '_train'] = []
            history[reg + '_val'] = []
            history_tags.append(reg)

    if metrics_dic is not None:
        for metric in metrics_dic.keys():
            history[metric + '_train'] = []
            history[metric + '_val'] = []
            history[metric + '_group_train'] = []
            history[metric + '_group_val'] = []

            history_tags.append(metric)
            history_tags.append(metric + '_group')

        if metric_stopper not in metrics_dic.keys():
            metric_stopper = 'loss'

    if eval_dataloader is not None:
        evaluation_list = []

    ## Add constrain for metric stopper
    if metric_stopper in metrics_dic.keys():
        history['constrain_' + metric_stopper + '_train'] = []
        history['constrain_' + metric_stopper + '_val'] = []
        history_tags.append('constrain_' + metric_stopper)
    else:
        metric_stopper = 'criterion'
        print('Metric stopper set to : ',metric_stopper)


    # stopper train and validation
    # metric_stopper_train = metric_stopper + '_train'
    # metric_stopper_val = metric_stopper + '_val'


    stop = False
    epoch = 0
    epoch_best = 0

    decay_patience = 0


    # prior = config.train_mode_params['group_prior']
    weights = config.train_mode_params['weights_init'] + 0
    weights = weights / np.sum(weights)



    # weight_list, lam_list, cost_list
    # lr_penalty = config.lr_penalty + 0


    if config.scheduler == 'OneCycleLR':
        scheduler_batch = scheduler
        print('scheduler batch :', scheduler_batch)
    else:
        scheduler_batch = None
    print('train num_max_batch : ',config.num_max_batch)
    while ((not stop) | (epoch < epoch_warmup)) & (epoch <= config.EPOCHS):

        # save learning rate
        learning_rate_all[epoch] = optimizer.param_groups[0]['lr'] + 0

        # weights = weights / np.sum(weights)

        history['weights_group'].append(weights)
        # history['lambda_group'].append(lambda_weights)

        # weights = weights
        output_train = epoch_training(train_dataloader, classifier_network, criterion, weights / group_prior,
                                      optimizer, config.DEVICE, metrics_dic=metrics_dic,
                                      train_type=True, reg_dic=reg_dic, reg_weights=reg_weights,
                                      scheduler=scheduler_batch, num_max_batch = config.num_max_batch)

        output_val = epoch_training(val_dataloader, classifier_network, criterion, weights / group_prior,
                                    optimizer, config.DEVICE, metrics_dic=metrics_dic,
                                    train_type=False, reg_dic=reg_dic, reg_weights=reg_weights)

        ## history update
        history['loss_train'].append(output_train['loss'])
        history['loss_val'].append(output_val['loss'])
        history['criterion_train'].append(output_train['criterion'])
        history['criterion_val'].append(output_val['criterion'])

        for group in range(ngroups):
            if group == 0:
                history['criterion_group_train'].append(np.zeros([ngroups]))
                history['criterion_group_val'].append(np.zeros([ngroups]))
            history['criterion_group_train'][-1][group] = output_train['criterion_group'][group]
            history['criterion_group_val'][-1][group] = output_val['criterion_group'][group]

            if metrics_dic is not None:
                for metric in metrics_dic.keys():
                    if group == 0:
                        history[metric + '_group_train'].append(np.zeros([ngroups]))
                        history[metric + '_group_val'].append(np.zeros([ngroups]))
                        history[metric + '_train'].append(output_train[metric])
                        history[metric + '_val'].append(output_val[metric])

                    history[metric + '_group_train'][-1][group] = output_train[metric + '_group'][group]
                    history[metric + '_group_val'][-1][group] = output_val[metric + '_group'][group]

        if reg_dic is not None:
            for reg in reg_dic.keys():
                history[reg + '_train'].append(output_train[reg])
                history[reg + '_val'].append(output_val[reg])

        ## save last model
        model_params_save(config.basedir + config.model_name + '/' + config.last_model,
                          classifier_network,
                          optimizer)

        ## Update - group constrain losses
        constrain_loss_train = history['criterion_group_train'][-1] - config.train_mode_params['group_constrain']
        constrain_loss_val = history['criterion_group_val'][-1] - config.train_mode_params['group_constrain']

        if config.train_mode_params['lr_penalty']>0: #lr_penalty = 0 indicates fix weights
            aux = (1-config.train_mode_params['min_weight']*constrain_loss_train.shape[0])*np.max(constrain_loss_train)
            aux += np.sum(config.train_mode_params['min_weight']*constrain_loss_train)
        else:
            aux = np.sum(weights*constrain_loss_train)
        history['constrain_train'].append(aux)

        if config.train_mode_params['lr_penalty']>0:
            aux = (1 - config.train_mode_params['min_weight'] * constrain_loss_val.shape[0]) * np.max(
                constrain_loss_val)
            aux += np.sum(config.train_mode_params['min_weight'] * constrain_loss_val)
        else:
            aux = np.sum(weights * constrain_loss_val)
        history['constrain_val'].append(aux)

        print('DEBUG TRAIN,VAL', constrain_loss_train, constrain_loss_val)
        print('DEBUG TRAIN,VAL',  history['constrain_train'][-1], history['constrain_val'][-1])


        ## metric for evaluation -> stopping criteria based on generalization to a metric constrain
        constrain_criteria_tag = 'constrain_'
        if metric_stopper != 'criterion':
            constrain_metric_train = history[metric_stopper + '_group_train'][-1] - config.train_mode_params['group_constrain_' + metric_stopper]
            constrain_metric_val = history[metric_stopper + '_group_val'][-1] - config.train_mode_params['group_constrain_' + metric_stopper]

            if metric_stopper in ['acc','softacc']:
                constrain_metric_train = -1 * constrain_metric_train
                constrain_metric_val = -1 * constrain_metric_val

            constrain_criteria_tag = constrain_criteria_tag + metric_stopper

            #worst case with the constrain of min weight
            if config.train_mode_params['lr_penalty']>0:
                aux = (1 - config.train_mode_params['min_weight'] * constrain_metric_train.shape[0]) * np.max(constrain_metric_train)
                aux += np.sum(config.train_mode_params['min_weight'] * constrain_metric_train)
            else:
                aux = np.sum(weights * constrain_metric_train)
            history['constrain_' + metric_stopper + '_train'].append(aux)

            if config.train_mode_params['lr_penalty'] > 0:
                aux = (1 - config.train_mode_params['min_weight'] * constrain_metric_val.shape[0]) * np.max(constrain_metric_val)
                aux += np.sum(config.train_mode_params['min_weight'] * constrain_metric_val)
            else:
                aux = np.sum(weights * constrain_metric_val)
            history['constrain_' + metric_stopper + '_val'].append(aux)

            print('DEBUG TRAIN,VAL', constrain_metric_train, constrain_metric_val)
            print('DEBUG TRAIN,VAL', history['constrain_' + metric_stopper + '_train'][-1], history['constrain_' + metric_stopper + '_val'][-1])

        #     ## second criteria
        #     second_criteria_val = np.mean(constrain_metric_val)
        #     second_criteria_train =np.mean(constrain_metric_train)
        #
        # else:
        #     ## second criteria
        #     second_criteria_val = np.mean(constrain_loss_val)
        #     second_criteria_train = np.mean(constrain_loss_train)

        ## for train we will consider loss constrain
        constrain_train = history['constrain_train'][-1]
        # constrain_val = history['constrain_val'][-1]

        ## for val we will consider generalization to the metric constrain
        # constrain_train = history[constrain_criteria_tag+'_train'][-1]
        constrain_val = history[constrain_criteria_tag+'_val'][-1]


        if (epoch == 0):

            # init stopper penalty -> train
            stopper_train = early_stopping(config.patience, 0, constrain_train) ## constrain is the condition to check
            epoch_best_train = epoch + 1
            model_params_save(config.basedir + config.model_name + '/' + config.best_model_train, classifier_network,
                              optimizer)

            # init stopper penalty -> val
            stopper = early_stopping(config.patience, 0, constrain_val) ## constrain is the condition to check
            epoch_best = epoch + 1
            model_params_save(config.basedir + config.model_name + '/' + config.best_model,
                              classifier_network,
                              optimizer)
        else:

            #save best train
            save_train, stop_train = stopper_train.evaluate(constrain_train)
            if (save_train):
                epoch_best_train = epoch + 1
                print('saving best train model, epoch: ', epoch_best_train)
                model_params_save(config.basedir + config.model_name + '/' + config.best_model_train,
                                  classifier_network,
                                  optimizer)

            #save best val
            save, stop = stopper.evaluate(constrain_val)
            if save :
                epoch_best = epoch + 1
                print('saving best model, epoch: ', epoch_best)
                model_params_save(config.basedir + config.model_name + '/' + config.best_model,
                                  classifier_network,
                                  optimizer)



        ##### print
        if ((epoch % config.n_print == config.n_print - 1) & (epoch >= 1)) | (epoch == 0):
            string_print = 'Epoch: ' + str(epoch) + '; lr: ' + str(np.round(optimizer.param_groups[0]['lr'],5))
            for tag in history_tags:
                string_print = string_print + '|' + tag + ' (tr,val): ' + str(
                    np.round(history[tag + '_train'][epoch], 3)) + \
                               ', ' + str(np.round(history[tag + '_val'][epoch], 3))

            string_print = string_print + '|stop_c : ' + str(stopper.counter) + '; best_train : ' +\
                           str(np.round(stopper_train.best_loss,4)) + ', epoch' + str(epoch_best_train)
            print(string_print + '; best_val : ' + str(np.round(stopper.best_loss,4)) + ', epoch' + str(epoch_best) )

            print()
            print(' constrain_loss_train : ' , constrain_loss_train)
            print(' constrain_loss_val : ', constrain_loss_val)
            if metric_stopper != 'criterion':
                print(' constrain_'+metric_stopper+ '_train : ', constrain_metric_train)
                print(' constrain_'+metric_stopper+ '_val : ', constrain_metric_val)
            print(' weights : ' , np.round(weights,4))

            # if config.dp:
            #     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(config.delta)
            #     print('DP epsilon: ', np.round(epsilon,2))


        # OPTIONAL: EVALUATION AND SAVING EVERY  'config.eval_every_epochs' or in the last epoch using best training model
        save_eval_flag = (eval_dataloader is not None) & \
                         (((epoch % config.eval_every_epochs == 0) & \
                           (config.eval_every_epochs > 0)) | stop | (epoch == config.EPOCHS))
        if save_eval_flag:
            print('**Saving Evaluation on epoch ', epoch)
            ## Evaluate:
            model_params_load(config.basedir + config.model_name + '/'+ config.best_model_train,
                              classifier_network,
                              None,
                              config.DEVICE)
            utility_pred_l, utility_gt_l, _ = fast_epoch_evaluation(eval_dataloader, classifier_network,
                                                                    config.n_utility, config.DEVICE,
                                                                    groups=True)
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

        if config.train_mode_params['lr_penalty']>0: # else we just keep the initial weights
            weight_list, cost_list = pga_weights(constrain_loss_train,
                                                 history['weights_group'][-1],
                                                 eta=config.train_mode_params['lr_penalty'],
                                                 iterations=2000,
                                                 cost_delta_improve=config.train_mode_params['cost_delta_improve'],
                                                 max_weight_change=config.train_mode_params['max_weight_change'],
                                                 e=config.train_mode_params['min_weight'])

            weights = weight_list[-1]
            if np.min(weights) < 0: ## TODO!: Check < epsilon (min_weight)
                print(' weights < 0 :', weights)
                weights = np.maximum(weights, 0)
            if np.abs(np.sum(weights) - 1) > 1e-5:
                print(' weights == :', np.sum(weights))
                weights = weights/np.sum(weights)
            print('New weights : ', weight_list[-1])

        if (scheduler is not None) &  (config.scheduler in ['CosineAnnealingLR','MultiStepLR']):
            print('scheduler step')
            scheduler.step()

        if (scheduler is not None) &  (config.scheduler in ['ManualLRDecayPlateau']):
            print('scheduler step ')
            scheduler.step(history['constrain_train'][-1])

        if (scheduler is not None) & (config.scheduler in ['ManualLRDecayNWReset']):
            print('scheduler step ')
            scheduler.step(constrain_train,
                           constrain_val)


        epoch += 1
        print()


    # -------- END TRAINING --------#

    # load best network
    print('Training Ended, loading best model : ', config.basedir + config.model_name + '/' + config.best_model)
    model_params_load(config.basedir + config.model_name + '/' + config.best_model, classifier_network, optimizer,
                      config.DEVICE)

    ## Eval best validation model ->
    if (eval_dataloader is not None):
        utility_pred_l, utility_gt_l, _ = fast_epoch_evaluation(eval_dataloader, classifier_network,
                                                                config.n_utility, config.DEVICE,
                                                                groups=True)
        utility_pred_l = np.array(utility_pred_l).transpose()
        columns = ['ypred_' + str(i) for i in range(utility_pred_l.shape[1])]
        pd_eval = pd.DataFrame(data=utility_pred_l, columns=columns)
        pd_eval['ygt'] = utility_gt_l
        pd_eval['epochs'] = epoch_best + 1
        evaluation_list.append(pd_eval)

        return history,evaluation_list
    else:
        return history



