import torch 
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix,  roc_auc_score
import shap
from set_variables import get_variables



v = get_variables()

spec_thresh = v.spec_thresh
use_thresh = v.use_thresh
get_shap = v.get_shap
model = v.model
nn_params = v.nn_params



def select_thresh(y_true_list, y_pos_probs):
    #Threshold selection
    if y_true_list != []:
        fpr, tpr, thresholds = roc_curve(y_true_list, y_pos_probs)

        #swtich around!!
        sensitivity = 1 - fpr
        specificity = tpr

    
        best_thresh = None
        best_sens = 0

        for i in range(len(thresholds)):
            if specificity[i]  >=spec_thresh and sensitivity[i] > best_sens:
                best_sens = tpr[i]
                best_thresh = thresholds[i]
    else: 
        best_thresh = None
    
    if best_thresh == None: 
        print("_"*30)
        print("Threshold selection failed")
        print("_"*30)
        
            
    return best_thresh


def train(_model, trainloader, epoch, fold, my_optimizer, loss_fn, y_true_list = [], y_pos_probs = [], writer = None):
    
    
    #####################################################
    #training
    #####################################################
    count = 0
    step_loss = []
    step_acc = []

    _model.train()
    for i, data in enumerate(trainloader): 
        X_batch, Y_batch = data
        #if epoch == 0 and i == 0:
            #writer.add_graph(_model, input_to_model=data[0], verbose=False)

        my_optimizer.zero_grad()
        Y_pred = _model(X_batch)

        loss = loss_fn(Y_pred, Y_batch)
        loss.backward()
        my_optimizer.step()

        fY_pred = torch.argmax(Y_pred, 1)
        
        acc = (fY_pred == Y_batch).float().sum()
        count += len(Y_batch)
        step_acc.append(acc)

        #collects these for threshold selection later
        pos_prob = Y_pred[:, 1]
        y_true = Y_batch.detach().numpy()
        y_pos_prob = pos_prob.detach().numpy()

        step_loss.append(loss.item())
      

    epoch_loss = np.array(step_loss).mean()
    epoch_acc = (np.array(step_acc).sum())/count

    if writer is not None:
        writer.add_scalar("Fold " + str(fold)+ '/Loss/Train', epoch_loss, epoch)
        writer.add_scalar("Fold " + str(fold)+ '/Accuracy/Train', epoch_acc, epoch)

    if use_thresh == True:
        best_thresh = select_thresh(y_true_list, y_pos_probs, spec_thresh)
    else:
        best_thresh = None


    return y_true, y_pos_prob, best_thresh, epoch_loss, epoch_acc

def val(_model, validloader, epoch, fold, loss_fn, mean_fpr, best_thresh, writer = None):

    #loads in state dict belonging to the best model
    #this is also the place to load in state dict of best model that was trained on a DIFFERENT antibiotic, for instance, to test generalizability.
    #_model.load_state_dict(torch.load(directory1))
    #dir = "D:/CNN_res/Kanamycin/95.03%_20x20_double_['_g0_']/best_model.pth"
    #dir = "D:/CNN_res/Trimethoprim/65.25%_20x20_double_['_g0_']/best_model.pth"
    #dir = "D:/CNN_res/Trimethoprim/66.40%_20x20_2DTIRF_['_g0_']/best_model.pth"
    ###i took above form prev test function


    #####################################################
    #validation/testing
    #####################################################
    val_count = 0
    val_step_loss = []
    val_step_acc = []

    y_pred_list = []
    y_true_list = []

    y_pos_prob = []
    X_list = []

    _model.eval()

    for X, Y in validloader: 
        with torch.no_grad(): 

            Y_pred = _model(X)

            val_loss = loss_fn(Y_pred, Y).item()
            val_step_loss.append(val_loss)
        

            fY_pred = torch.argmax(Y_pred, 1)

            val_acc = (fY_pred == Y).float().sum()
            val_count += len(Y)
            val_step_acc.append(val_acc)

    
            pos_prob = Y_pred[:, 1]
            
            y_pred_list.extend(fY_pred)
            y_true_list.extend(Y)
            y_pos_prob.extend(pos_prob)
            X_list.extend(X)


    val_epoch_loss = np.array(val_step_loss).mean()
    val_epoch_acc = (np.array(val_step_acc).sum())/val_count

   
    if writer is not None:

        writer.add_scalar("Fold " + str(fold)+ '/Loss/Test', val_epoch_loss, epoch)

        writer.add_scalar("Fold " + str(fold)+ '/Accuracy/Test', val_epoch_acc, epoch)

        writer.add_histogram("Fold " + str(fold) + "/Linear1.bias", _model.layer1[0].bias, epoch)
        writer.add_histogram("Fold " + str(fold)+"/Linear1.weight", _model.layer1[0].weight, epoch)
        writer.add_histogram("Fold " + str(fold)+"/Linear_Out.bias", _model.output[1].bias, epoch)
        writer.add_histogram("Fold " + str(fold)+ "/Linear_Out.weight", _model.output[1].weight, epoch)

    #ROC
    fpr, tpr, thresholds = roc_curve(y_true_list, y_pos_prob)
    roc_auc = roc_auc_score(y_true_list, y_pos_prob)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    
    if best_thresh is not None:
        y_pred_list = (y_pos_prob >= best_thresh).astype(int)
    else:
        y_pred_list = y_pred_list

    cm = confusion_matrix(y_true_list, y_pred_list)

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    #sensitivity (true positive rate) and specificity (true negative rate):  take positive = class 1, negative = class 0

    sens = tp/(tp+fn)
    spec = tn/(fp+tn)
    g_mean = np.sqrt(sens*spec)


    precision = tp/(tp+fp)
    recall = sens
    f1 = 2*((precision*recall)/(precision+recall))

    
    #SHAP analysis
    if get_shap == True and model not in ["CNN_LSTM", "LSTM"] and len(nn_params) == 1:

        batch = next(iter(validloader))
        images, labels = batch

        batchsize = len(images)

        background = images[:(batchsize-2)]
        test_images = images[(batchsize -2):batchsize]
        test_labels = labels[(batchsize -2):batchsize]
        

        e = shap.DeepExplainer(_model, background)
        shap_values = e.shap_values(test_images, check_additivity = False )
        base_val = e.expected_value

        shap_values = np.swapaxes(shap_values, 1, -1)
        shap_values = np.swapaxes(shap_values, 0, 1)

        shap_numpy = [s for s in shap_values]

        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)


        #shap_numpy = shap_values
        #test_numpy = test_images

        p = shap_numpy, test_numpy, test_labels, base_val
    
    else: 
        p = 0

    return val_epoch_acc, val_epoch_loss, sens, spec, g_mean, f1, roc_auc, interp_tpr, y_pred_list, y_true_list, p, 

def train_test(fold, _model, trainloader, validloader, trainvalloader, testloader, max_epochs, loss_fn, my_optimizer, early_stop_thresh, dir_mod, mean_fpr, validation = False, writer = None):  

    #this is for the Early Stopping in validation
    best_accuracy = -1
    best_epoch = -1

    y_true_list = []
    y_pos_probs = []

    trainingEpoch_loss = []
    trainingEpoch_acc = []

    testEpoch_loss = []
    testEpoch_acc = []

    for epoch in range(max_epochs):
        
        
        if validation == True:
            y_true, y_pos_prob, best_thresh, tr_loss, tr_acc = train(_model, trainloader, epoch, fold, my_optimizer, loss_fn, y_true_list, y_pos_probs, writer)
            y_true_list.extend(y_true)
            y_pos_probs.extend(y_pos_prob)
         

            val_epoch_acc, _, _,  _, _, _, _, _, _, _, _,  = val(_model, validloader, epoch, fold, loss_fn, mean_fpr, best_thresh, writer)

        elif validation == False: 
            y_true, y_pos_prob, best_thresh, tr_loss, tr_acc = train(_model, trainvalloader, epoch, fold, my_optimizer, loss_fn, spec_thresh, y_true_list, y_pos_probs, writer, use_thresh)
            y_true_list.extend(y_true)
            y_pos_probs.extend(y_pos_prob)
            
      
        
        torch.save(_model.state_dict(), dir_mod + "best_model.pth")

    
        ts_acc, ts_loss, sens, spec, g_mean, f1, roc_auc, interp_tpr, y_pred_list_test, y_true_list_test, p = val(_model, testloader, epoch, fold, loss_fn, mean_fpr, best_thresh,  writer)
        print("Epoch %d: test loss %.2f" % (epoch, ts_loss))
        print("Epoch %d: test accuracy %.2f%%" % (epoch, ts_acc*100))
    
        trainingEpoch_loss.append(tr_loss)
        trainingEpoch_acc.append(tr_acc)

        testEpoch_loss.append(ts_loss)
        testEpoch_acc.append(ts_acc)
        #####################################################
        #Early Stopping - only if an additional validation set is used
        if validation == True:

            if val_epoch_acc > best_accuracy:
                best_accuracy = val_epoch_acc
                best_epoch = epoch

            #Early stopping
            elif epoch - best_epoch > early_stop_thresh:
                print("Early stopped training at epoch %d" % epoch)
                break  # terminate the training loop

        #####################################################

        
        
        np.save(dir_mod + "best_thresh", np.array(best_thresh))
        
    return  sens, spec, g_mean, f1, roc_auc, interp_tpr, y_pred_list_test, y_true_list_test, p, trainingEpoch_loss, trainingEpoch_acc,testEpoch_loss, testEpoch_acc