import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import time
import shutil
import os
import shap
import torch
from helper_functions import initialise
from set_variables import get_variables



v = get_variables()
model = v.model
nn_params = v.nn_params
antibiotic = v.antibiotic
keyword = v.keyword
method = v.method
save_res = v.save_res
fit = v.fit
save_splits = v.save_splits
get_shap = v.get_shap

def plot_epochs(directory, trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc): 

    fig, axes= plt.subplots(2, 1, figsize=(10, 8), sharex = False)
    axes[0].plot(trainingEpoch_loss, label='train_loss')
    axes[0].plot(valEpoch_loss,label='val_loss')
    axes[0].legend()

    axes[1].plot(trainingEpoch_acc, label='train_accuracy')
    axes[1].plot(valEpoch_acc,label='val_accuracy')
    axes[1].legend()
    
    plt.suptitle("Accuracy and Loss over epochs (last fold)")
    plt.savefig(directory, dpi = 300)

def shap_plot(shapplot):

    if model not in ["CNN_LSTM", "LSTM"]:


    
        #SHAP plotting
        shapnp, testnp, testlabels, baseval = shapplot
        t = shapnp[0]

        """
        #this is for 300x4 plotting - need to save manually

        threshold_low = -0.006
        threshold_high = 0.006

        blue = (0, 0, 1)     
        white = (1, 1, 1) 
        red = (1, 0, 0)      

    
        n_bins = 100  
        cmap_name = 'blue_gray_red'
        cmap = colors.ListedColormap([ blue, white, red])
       
        bounds = [-1, threshold_low, threshold_high, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        fig = plt.figure(figsize=(4, 4), dpi=300)
        gs = fig.add_gridspec(4, 1)


        fig.add_subplot(gs[0, 0])
        plt.imshow(testnp[0], cmap = "Greys_r")
        plt.tick_params(
                    bottom=False,
                    left = False,       
                    labelbottom=False, 
                    labelleft = False) 

    
        fig.add_subplot(gs[1, 0])
        plt.imshow(shapnp[0][0],cmap=cmap, norm=norm, interpolation='nearest')
        plt.tick_params(
                    bottom=False,
                    left = False,       
                    labelbottom=False, 
                    labelleft = False) 

      
        fig.add_subplot(gs[2, 0])
        plt.imshow(testnp[1], cmap = "Greys_r")
        plt.tick_params(
                    bottom=False,
                    left = False,       
                    labelbottom=False, 
                    labelleft = False) 

      
        fig.add_subplot(gs[3, 0])
        plt.imshow(shapnp[0][1], cmap=cmap, norm=norm, interpolation='nearest')
        plt.tick_params(
                    bottom=False,
                    left = False,       
                    labelbottom=False, 
                    labelleft = False) 

        #cbar = plt.colorbar(orientation='vertical')
        #cbar.ax.tick_params(labelsize=5)
          

        oglabels = ["C" + str(testlabels[0].item()), "C" + str(testlabels[1].item())]
        ogx_position = 0.08
        ogy_positions = [0.78, 0.4]  

        for y_pos, label in zip(ogy_positions, oglabels):
            plt.text( ogx_position, y_pos, label, transform=plt.gcf().transFigure, va='center', fontsize=5)



        plt.subplots_adjust(hspace=0.001) 
        plt.tight_layout()
       
        plt.show()
        
        """
       
    
        #This is for when have 2 TEST IMAGES
        #fig = plt.figure()
        shap.image_plot(shapnp, testnp, show = False)
        labels = ["Original Im" + str(nn_params), "Class 0 SHAP", "Class 1 SHAP"]
        x_positions = [0.235, 0.49, 0.76]  
        y_position = 0.93

        oglabels = ["Class " + str(testlabels[0].item()), "Class " + str(testlabels[1].item())]
        ogx_position = 0.15
        ogy_positions = [0.9, 0.6]  

        for y_pos, label in zip(ogy_positions, oglabels):
            plt.text( ogx_position, y_pos, label, transform=plt.gcf().transFigure, va='center', fontsize=12)

        for x_pos, label in zip(x_positions, labels):
            plt.text(x_pos, y_position, label, transform=plt.gcf().transFigure, ha='center', fontsize=12)

        
        return plt.gcf()

def shap_flat_plot(shaplist, directory2): 

        #have shap values for class 0 and for class 1
        #since binary, all info is contained in one. eg shap0 positive contribute to class 0, negative contribute to class 1

    shap0 = []
    testims = []
    for el in shaplist: 
        shapnp, testnp, testlabels, base_val = el
        shapclass0 = shapnp[0].flatten()
        shap0.extend(shapclass0)
        testims.extend(testnp.flatten())

    np.save(directory2 + "flat shap vs param data", np.array([shap0, testims]))

    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(shap0, testims, c = "b", alpha = 0.1)
    plt.xlabel("SHAP value")
    plt.ylabel(nn_params)
    plt.grid()
    plt.savefig(directory2 + "paramvsshap",dpi=300)
    #plt.show()

def plot_featuremaps(dir_mod,  testloader ):
                        
    fmaps_dir = dir_mod + "featuremaps"

    newmodel, batchsize = initialise()

    #loads in state dict belonging to the best model
    newmodel.load_state_dict(torch.load(dir_mod + "best_model.pth"))
    newmodel.eval()

    for X, Y in testloader: 
        X = X.to(torch.float32)
        for j in range(len(X)): 
            x = X[j]
            y = Y[j]
            

            if "ACF" in nn_params: 
                pass

            else: 
                plt.figure()
                plt.imshow(x[0])
                plt.savefig(fmaps_dir  + "/" + str(j) + "_im_class" + str(y.item()), dpi = 300) 
                with torch.no_grad(): 
                    feature_maps = newmodel.layer1(x)
                #fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))

                if "300x4" in keyword:
                    fig = plt.figure(figsize=(4, 32))
                    gs = fig.add_gridspec(32, 1)
                    for i in range(32):
                        ax = fig.add_subplot(gs[i, 0])
                        ax.imshow(feature_maps[i])
                        ax.axis("off")

                    plt.tight_layout()  
                    plt.savefig(fmaps_dir  + "/" + str(j) + "_1stConvlayer_class" + str(y.item()))
                    plt.show()

                else:
                    fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
                    plt.axis("off")
                    for i in range(0, 32):
                        row, col = i//8, i%8
                        ax[row][col].imshow(feature_maps[i])
                    plt.savefig(fmaps_dir  + "/" + str(j) + "_1stConvlayer_class" + str(y.item()) , dpi = 300)

            
            """
            with torch.no_grad(): 
                feature_maps = newmodel.layer2(feature_maps)

            fig2, ax2 = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
            for j in range(0, 32):
                row, col = j//8, j%8
                ax2[row][col].imshow(feature_maps[j])
            plt.savefig(directory3 + "/3rdConvlayer_" + str(i), dpi = 300)

            with torch.no_grad(): 
                feature_maps = newmodel.layer3(feature_maps)

            fig2, ax2 = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
            for j in range(0, 32):
                row, col = j//8, j%8
                ax2[row][col].imshow(feature_maps[j])
            plt.savefig(directory3 + "/3rdConvlayer_" + str(i), dpi = 300)
        """   
        break

def fold_res(k_folds, fold_metrics, tprs, y_true_folds, y_pred_folds,  mean_fpr, trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc, shap_folds_list, shapfigs, dir_mod):


    # Print fold results
    print('----------------------------------------------------')
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('----------------------------------------------------')
  
    fold_test_accs = []
    fold_aucs = []
    fold_specs = []
    fold_sens = []
    fold_gmeans = []
    fold_f1s = []

    #SEE if instead I can just turn fold_metric into a DF and then take means. instead of having to do the list thing.
    #cause then can also save as csv later  instead of having to write a text file

    for m in fold_metrics: 

        fold_test_accs.append(fold_metrics[m]["test accuracy"])
        fold_aucs.append(fold_metrics[m]["auc"])
        fold_specs.append(fold_metrics[m]["spec"])
        fold_sens.append(fold_metrics[m]["sens"])
        fold_gmeans.append(fold_metrics[m]["gmean"])
        fold_f1s.append(fold_metrics[m]["f1"])

        test_acc = fold_metrics[m]["test accuracy"]
        test_sens = fold_metrics[m]["sens"]
        test_spec = fold_metrics[m]["spec"]
        print("Fold %d: Acc: %.4f Sens: %.4f Spec: %.4f" % (m, test_acc, test_sens, test_spec))

    mean_test_acc = np.mean(fold_test_accs)
    sd_test_acc = np.std(fold_test_accs)

    mean_spec = np.mean(fold_specs)
    sd_spec = np.std(fold_specs)

    mean_sens = np.mean(fold_sens)
    sd_sens = np.std(fold_sens)

    mean_gmean = np.mean(fold_gmeans)
    sd_gmean = np.std(fold_gmeans)

    mean_auc = np.mean(fold_aucs)
    sd_auc = np.std(fold_aucs)

    mean_tpr = np.mean(tprs, axis=0)
    sd_tpr = np.std(tprs, axis = 0)
    
    mean_f1 = np.mean(fold_f1s)
    sd_f1 = np.std(fold_f1s)
    
    print("Mean test accuracy: %.4f \u00B1 %.4f" % (mean_test_acc, sd_test_acc))
    print("Mean specificity: %.4f \u00B1 %.4f" % (mean_spec, sd_spec))
    print("Mean sensitivity: %.4f \u00B1 %.4f" % (mean_sens, sd_sens))
    print("Mean G-mean: %.4f \u00B1 %.4f" % (mean_gmean, sd_gmean))
    print("Mean AUC: %.4f \u00B1 %.4f" %  (mean_auc, sd_auc))
    print("Mean F1: %.4f \u00B1 %.4f" % (mean_f1, sd_f1))


    report = classification_report(y_true_folds, y_pred_folds )
    print(report)
    #directory2 = "D:/" + model + "_res/" + antibiotic + "/" + "%.2f%%" % ((mean_test_acc*100)) + "_" + ROI + "_" + fit  + "_" + str(nn_params) + "/"

    #if os.path.exists(directory2) == False: 
        #os.makedirs(directory2)

    #plots training and validaton acc and loss over epochs, for the last fold
    #(doesn't make sense to plot mean curves because different epochs for each fold due to early stopping)
    if save_res == True: 
        
        timenow = time.localtime()
        time_str = ""
        for el in timenow[:6]: 
            time_str = time_str + str(el) + "_"

        result_dir = "D:/" + model + "_res/" + antibiotic + "/" + "%.2f%%" % ((mean_test_acc*100)) +  "_" + keyword + "_" + fit  + "_" + str(nn_params) + "_" + time_str + "/"

        if os.path.exists(result_dir) == False: 
                os.makedirs(result_dir)

        plot_epochs(result_dir+ "metrics_plot", trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc)

        #get confusion matrix of all folds
        cm = confusion_matrix(y_true_folds, y_pred_folds)
        cmfig = ConfusionMatrixDisplay(cm)
        cmfig.plot()
        plt.title("Confusion matrix of 5 folds")
        cmfig.figure_.savefig(result_dir+ "conf_mx.png",dpi=300)

        #plot Mean ROC curve with standard deviations
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=("Mean ROC (AUC %.2f $\pm$ %.2f)" %  (mean_auc, sd_auc)),
            lw=2,
            alpha=0.8,
        )

        ax.plot([0, 1], ls="--")

        tprs_upper = np.minimum(mean_tpr + sd_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - sd_tpr, 0)

        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve",
        )
        ax.legend(loc="lower right")
        plt.savefig(result_dir + "roc.png",dpi=300)
        
        rc = np.array([mean_tpr, tprs_upper, tprs_lower])
        a = np.array([mean_auc, sd_auc])
        np.save(result_dir + "mean roc curve data.npy", rc)
        np.save( result_dir+ "mean auc data.npy", a)
                    

        shutil.move(dir_mod + "best_model.pth", result_dir + "best_model.pth")
        shutil.move(dir_mod + "best_thresh.npy", result_dir + "best_thresh.npy")

        if save_splits == True:
            shutil.move(dir_mod + "splits.json", result_dir + "splits.json")

        
        fold_all_res = pd.DataFrame()
        fold_all_res["fold_test_accuracies"] = [np.round(el, 4) for el in fold_test_accs] + [np.round(mean_test_acc, 4), np.round(sd_test_acc, 4)]
        fold_all_res["fold_aucs"] = [np.round(el, 4) for el in fold_aucs]  +  [np.round(mean_auc, 4), np.round(sd_auc)]
        fold_all_res["fold_specificities"] = [np.round(el, 4) for el in fold_specs] +  [np.round(mean_spec, 4), np.round(sd_spec, 4)]
        fold_all_res["fold_sensitivities"] = [np.round(el, 4) for el in fold_sens] +  [np.round(mean_sens, 4), np.round(sd_sens, 4)]
        fold_all_res["fold_g_means"] = [np.round(el, 4) for el in fold_gmeans] +  [np.round(mean_gmean, 4), np.round(sd_gmean, 4)]
        fold_all_res["fold_f1s"] = [np.round(el, 4) for el in fold_f1s] +  [np.round(mean_f1, 4), np.round(sd_f1, 4)]
        fold_all_res.index = list(np.arange(len(fold_test_accs))) + ["Mean", "Uncertainty"]
        fold_all_res.to_csv(result_dir + "fold_result_metrics")
        

        if get_shap == True and model not in ["CNN_LSTM", "LSTM"] and (method not in ["acf-1DNN"]) and len(nn_params) == 1:
            i = 0
            for fig in shapfigs: 
                fig.savefig(result_dir + "shap" + str(i))
                i += 1

            shap_flat_plot(shap_folds_list, result_dir)
    
    else: 
        os.remove(dir_mod + "best_model.pth")
        os.remove(dir_mod + "best_thresh.npy")
        os.remove(dir_mod + "splits.json")
        