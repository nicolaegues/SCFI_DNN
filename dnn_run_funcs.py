
import torch
from torch import nn
import numpy as np
import json
from sklearn.model_selection import  StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler



from helper_functions import CustomDataset, initialise, get_val_splits, get_summary, get_non_sampling_loaders
from training_testing import train_test
from set_variables import get_variables, Hyperparameters
from results import shap_plot, fold_res, plot_featuremaps
from dnn_models import CNN_Model


v = get_variables()
nn_params = v.nn_params
cross_val = v.cross_val
antibiotic = v.antibiotic
model = v.model
get_splits_i = v.get_splits_i
get_shap = v.get_shap
save_splits = v.save_splits
validation = v.validation
keyword = v.keyword


hp = Hyperparameters()
lstm_A_lr= hp.lstm_A_lr
lstm_early_stop_thresh = hp.lstm_early_stop_thresh
lstm_max_epochs = hp.lstm_max_epochs
cnn_A_lr = hp.cnn_A_lr
cnn_early_stop_thresh = hp.cnn_early_stop_thresh
cnn_max_epochs = hp.cnn_max_epochs
optimizer = hp.optimizer
loss_fn = hp.loss_fn
k_folds = hp.k_folds

def feature_extraction(total_X_arr, G_total_X_tr_arr, G_total_X_ts_arr): 

    nr_params = total_X_arr.shape[1]

    feature_extractor = CNN_Model(nr_params, keyword)

    if "timeseries" in nn_params: 
        #feature_extractor.load_state_dict(torch.load("D:/CNN_res/Trimethoprim/47.55%_20x20_none_['timeseries']REP/best_model.pth"))
        #feature_extractor.load_state_dict(torch.load("D:/CNN_res/Kanamycin/76.40%_20x20_none_['timeseries']REP/best_model.pth"))
        #feature_extractor.load_state_dict(torch.load( "D:/CNN_res/Gentamycin/66.73%_20x20_none_['timeseries']/best_model.pth"))
        
        feature_extractor.load_state_dict(torch.load("D:/CNN_res/Gentamycin/97.41%_20x20_none_['timeseries']/best_model.pth"))
        #feature_extractor.load_state_dict(torch.load("D:/CNN_res/Kanamycin/99.97%_20x20_none_['timeseries']/best_model.pth"))
        #feature_extractor.load_state_dict(torch.load("D:/CNN_res/Trimethoprim/96.99%_20x20_none_['timeseries']/best_model.pth"))
        
    if cross_val == False and antibiotic == "Gentamycin": 
        # global total_X_tr_arr
        # global total_X_ts_arr
        total_X_array = np.concatenate((G_total_X_tr_arr, G_total_X_ts_arr), axis=0)

    else: 
        total_X_array = total_X_arr 

    vid_nr_frames = total_X_array.shape[2]

    frame_features = np.zeros(shape=(len(total_X_array), vid_nr_frames, 128), dtype="float32")

    for vid in range(len(total_X_array)): 
        temp_frame_features = np.zeros(shape=( vid_nr_frames, 1, 128), dtype="float32")

        
        frames = total_X_array[vid]
        frames = frames.transpose(1, 0, 2, 3)
        frames = frames[None, ...]
        frames = torch.tensor(frames)
        
        
        for i, batch in enumerate(frames):
            for frame in range(vid_nr_frames): 
                #transform = ToTensor()
                #input = transform(batch[None, frame, :])
                input = batch[None, frame, :]
                res = feature_extractor(input)
                res = res.detach().numpy()
                temp_frame_features[frame, :] = res

        frame_features[vid,] = temp_frame_features.squeeze()

    new_total_X_arr = frame_features
    np.save("D:/train_test arrays/"+antibiotic +"/Xfeature_arr_" + antibiotic + "_" + str(nn_params), new_total_X_arr)

    return new_total_X_arr

def one_fold_run(fold, my_model, dataset, batch_size, trainloader, validloader, trainvalloader, testloader, max_epochs, loss_fn, my_optimizer, early_stop_thresh, dir_mod, mean_fpr, get_featuremaps ):
    
        #get model architecture summary
        if fold == 0:
            get_summary(my_model, dataset, batch_size)


        print(f"Fold {fold + 1}")
        print("-------")

        
        sens, spec, g_mean, f1, roc_auc, interp_tpr, y_pred_list, y_true_list, shapplot, trainingEpoch_loss, trainingEpoch_acc, testEpoch_loss, testEpoch_acc = train_test(fold, my_model, trainloader, validloader, trainvalloader, testloader, max_epochs, loss_fn, my_optimizer, early_stop_thresh, dir_mod, mean_fpr, validation = validation, writer = None)


        #placeholders as no featuremaps and no shap (yet) for LSTMs

        if get_shap == True and len(nn_params) == 1 and model == "CNN":
            fig = shap_plot(shapplot)
            shapfigs.append(fig)
            shap_folds_list.append(shapplot)

        else: 
            #placeholder as can't do SHAP with more than one parameter 
            shap_folds_list = []
            shapfigs = 0
        

        if get_featuremaps == True:

                #only gets featuremaps for first fold
                if fold == 0:
                    plot_featuremaps(dir_mod, testloader)
        
        return sens, spec, g_mean, f1, roc_auc, interp_tpr, y_pred_list, y_true_list, trainingEpoch_loss, trainingEpoch_acc, testEpoch_loss, testEpoch_acc
                    

def run_CNN_LSTM_models(total_X_arr, total_Y_arr, G_total_X_tr_arr, G_total_X_ts_arr, G_total_Y_tr_arr, G_total_Y_ts_arr, get_featuremaps = False): 


    ########################## Feature extraction for CNN_LSTM ##########################
    

    if model == "CNN_LSTM": 

        print("Running feature extraction")

        new_total_X_arr = feature_extraction(total_X_arr, G_total_X_tr_arr, G_total_X_ts_arr)

    ################################# DNN #################################

    #new_total_X_arr = np.load("D:/train_test arrays/"+antibiotic +"/Xfeature_arr_" + antibiotic + "_" + str(nn_params) + ".npy")
        if cross_val == False and antibiotic == "Gentamycin": 
            G_total_X_tr_arr = new_total_X_arr[:int(len(new_total_X_arr)*2/3)]
            G_total_X_ts_arr = new_total_X_arr[-int(len(new_total_X_arr)*1/3):]
        
        else: 
            total_X_arr = new_total_X_arr
    
    # elif model != "CNN_LSTM": 
    #     if not (cross_val == False and antibiotic == "Gentamycin"): 
    #         X_arr = total_X_arr


    #hyperparameters
    if model in ["CNN_LSTM","LSTM"]:
        print("Running LSTM")
        A_lr= lstm_A_lr
        early_stop_thresh = lstm_early_stop_thresh
        max_epochs = lstm_max_epochs

    else: 
        print("Running CNN")
        A_lr = cnn_A_lr
        early_stop_thresh = cnn_early_stop_thresh#5normal
        max_epochs = cnn_max_epochs #30 normal


    if cross_val == False and antibiotic == "Gentamycin":
        G_total_Y_tr = torch.LongTensor(G_total_Y_tr_arr)
        G_total_Y_ts = torch.LongTensor(G_total_Y_ts_arr)
        dataset = CustomDataset(G_total_X_tr_arr, G_total_Y_tr)

        nr_params = G_total_X_tr_arr.shape[1]
        seq_length = G_total_X_tr_arr.shape[2]

    else:
        total_Y = torch.LongTensor(total_Y_arr)
        #dataset = CustomDataset(X_arr, total_Y)
        dataset = CustomDataset(total_X_arr, total_Y)
        nr_params = total_X_arr.shape[1]
        seq_length = total_X_arr.shape[2]

    #the sequence length (length sequence inputted into LSTM) in this case is the lenght of the feature vector
    #(but isn't this just for i-CNNLSTM? what about the other timeseries methods?)
    if "timeseries" in nn_params:
        seq_length = 128
    
          
    dir_mod = "D:/" + model + "_res/" + antibiotic +  "/"


    #fold metrics
    fold_metrics = {}
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    shapfigs = []
    shap_folds_list = []

    y_true_folds = []
    y_pred_folds = []

    if cross_val == True:

        #Define the K-fold Cross Validator - stratified so same percentage of classes in each split
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
        fold_splits = {}

        for fold, (trainval_idx, test_idx) in enumerate(kfold.split(dataset, dataset.Y)):

            #Initialise model for every fold (otherwise will have learned)
            my_model, batch_size = initialise(model, nr_params, seq_length, keyword)

            my_optimizer = optimizer(my_model.parameters(), lr=A_lr)

            if get_splits_i == True:

                #this is if further splitting of the trainval into train and validation wants to be done.
                train_idx, val_idx = get_val_splits(trainval_idx,  dataset)

                #this is to later save the splits if wanted
                fold_splits[str(fold) + "_trainval"] = trainval_idx.tolist()
                fold_splits[str(fold) + "_train"] = train_idx.tolist()
                fold_splits[str(fold) + "_val"] = val_idx.tolist()
                fold_splits[str(fold) + "_test"] = test_idx.tolist()

            elif get_splits_i == False:

                splits = open("/splits.json")
                splits = json.load(splits)

                trainval_idx = splits[str(fold) + "_trainval"]
                train_idx = splits[str(fold) + "_train"]
                val_idx = splits[str(fold) + "_val"]
                test_idx = splits[str(fold) + "_test"] 

            trainvalloader  = DataLoader(dataset, batch_size=batch_size, sampler = SubsetRandomSampler(trainval_idx))
            trainloader  = DataLoader(dataset, batch_size=batch_size, sampler = SubsetRandomSampler(train_idx))
            validloader  = DataLoader(dataset, batch_size=batch_size, sampler = SubsetRandomSampler(val_idx))
            testloader  = DataLoader(dataset,  batch_size=batch_size, sampler = SubsetRandomSampler(test_idx))


            sens, spec, g_mean, f1, roc_auc, interp_tpr, y_pred_list, y_true_list, trainingEpoch_loss, trainingEpoch_acc, testEpoch_loss, testEpoch_acc = one_fold_run(fold, my_model, dataset, batch_size, trainloader, validloader, trainvalloader, testloader, max_epochs, 
                                                                                                                                                                       loss_fn, my_optimizer, early_stop_thresh, dir_mod, mean_fpr, get_featuremaps)
            y_pred_folds.extend(y_pred_list)
            y_true_folds.extend(y_true_list)
            tprs.append(interp_tpr)
            test_acc = testEpoch_acc[-1]
            fold_metrics[fold] = {"test accuracy": test_acc.item(), "sens": sens, "spec": spec.item(), "gmean": g_mean, "auc": roc_auc, "f1": f1}

        if save_splits == True:
            with open(dir_mod + "splits.json", 'w') as file:
                json.dump(fold_splits, file)
        
    
    elif cross_val == False: 

        for fold in range(3):
            
                   
            my_model, batch_size = initialise(model, nr_params, seq_length, keyword)
            my_optimizer = optimizer(my_model.parameters(), lr=A_lr)


            if antibiotic == "Gentamycin": 
                X_trainval = G_total_X_tr_arr
                Y_trainval = G_total_Y_tr

                X_test = G_total_X_ts_arr
                Y_test = G_total_Y_ts
            else: 
                #X_trainval = X_arr[:int(len(total_Y)*2/3)]
                X_trainval = total_X_arr[:int(len(total_Y)*2/3)]
                Y_trainval = total_Y[:int(len(total_Y)*2/3)]

                #X_test = X_arr[int(len(total_Y)*2/3): int(len(total_Y))]
                X_test = total_X_arr[int(len(total_Y)*2/3): int(len(total_Y))]
                Y_test = total_Y[int(len(total_Y)*2/3):int(len(total_Y))]

            trainloader, validloader, testloader, trainvalloader = get_non_sampling_loaders(X_trainval, Y_trainval, X_test, Y_test, batch_size)

            
            sens, spec, g_mean, f1, roc_auc, interp_tpr, y_pred_list, y_true_list, trainingEpoch_loss, trainingEpoch_acc, testEpoch_loss, testEpoch_acc = one_fold_run(fold, my_model, dataset, batch_size, trainloader, validloader, trainvalloader, testloader, max_epochs, 
                                                                                                                                                                       loss_fn, my_optimizer, early_stop_thresh, dir_mod, mean_fpr, get_featuremaps)            
            y_pred_folds.extend(y_pred_list)
            y_true_folds.extend(y_true_list)
            tprs.append(interp_tpr)
            test_acc = testEpoch_acc[-1]
            fold_metrics[fold] = {"test accuracy": test_acc.item(), "sens": sens, "spec": spec.item(), "gmean": g_mean, "auc": roc_auc, "f1": f1}
       

    fold_res(k_folds, fold_metrics, tprs, y_true_folds, y_pred_folds,  mean_fpr, trainingEpoch_loss, trainingEpoch_acc, testEpoch_loss, testEpoch_acc,  shap_folds_list, shapfigs, dir_mod)
