
from helper_functions import res_sus_distinguisher
import torch

#directory where the Saved arrays are
directory = "D:/Saved arrays"


gentamycin_folders= ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
kanamycin_folders = ["fig4 kanamycin1", "fig4 kanamycin2", "fig4 kanamycin3"]
trimethoprim_folders =  ["S3", "S4", "S5"]

#Gentamycin 01-10 correspond to these MIC values (in this order): 1, 1, 4, 16, 32, 2, 8, 64, 4, 8 (µg/ml)
#whereby susceptible < MIC 16 <= resistant
genta_y = [0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
kana_res_sus, kana_y = res_sus_distinguisher(kanamycin_folders)
trime_res_sus, trime_y = res_sus_distinguisher(trimethoprim_folders)


def_antibiotic = "Trimethoprim"
def_method = "acf-LSTM"
def_fit = "double"
def_nn_params = ["ACF"]
def_keyword = "BIN"


class Variables(): 
    def __init__(self, antibiotic, keyword, method, fit, nn_params, model, flatten, split_ACF, split_timeseries, ROI):
        
        self.data_directory = "D:/Saved arrays"
        self.spreadsheets_dir = "D:\Data spreadsheets"

        self.antibiotics_dict = {
            "Gentamycin": {"folders": gentamycin_folders, "y_vals": genta_y},
            "Kanamycin": {"folders": kana_res_sus, "y_vals": kana_y},
            "Trimethoprim": {"folders": trime_res_sus, "y_vals": trime_y},
        }

        self.keyword = keyword
        self.antibiotic = antibiotic
        self.method = method
        self.fit = fit
        self.nn_params = nn_params

        self.model = model
        self.flatten = flatten
        self.split_ACF = split_ACF
        self.split_timeseries = split_timeseries
        self.ROI = ROI

        self.spec_thresh = 0.5
        self.use_thresh = False

        self.equal_Genta = True
        self.cross_val = False

        self.validation = True
        self.get_shap = True
        self.get_splits_i = True
        self.save_splits = False
        self.save_res = True
        self.log_tensorboard = False # IMPLEMENT



class Hyperparameters(): 
    def __init__(self):
    
        self.lstm_A_lr=0.005
        self.lstm_early_stop_thresh = 10
        self.lstm_max_epochs = 100

        self.cnn_A_lr = 0.0005
        self.cnn_early_stop_thresh = 10#5normal
        self.cnn_max_epochs = 50 #30 normal

        self.optimizer = torch.optim.Adam()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.k_folds = 5

##############################################################################################################3


def input_params(fit, method):
    print("Note: g0 and intensities don't vary across fits, the options are just given for each anyway. ")

    while True: 
        if fit == "single": 
            nn_params = input('Choose one or more of the following, separated by a comma w/0 space: "_m_mono_", "t_mono", "R", "_g0_", "intensities"').split(",")

        elif fit == "double":
            nn_params = input('Choose one or more of the following, spearated by a comma w/o space: "_m1_", "_m2_", "t1", "t2", "_g0_", "intensities"').split(",")

        elif fit == "3DTIRF":
            nn_params = input('Choose one or more of the following, separated by a comma w/o space:  "N", "kappa", "tD", "_g0_", "intensities"').split(",")
        
        elif fit == "2DTIRF":
            nn_params = input('Choose one or more of the following, separated by a comma w/o space:  "N_2D", "tD_2D", "_g0_", "intensities"').split(",")

        elif fit == "2Dcutsigma":
            nn_params = input('Choose one or more of the following, separated by a comma w/o space:  "N_2Dcutsigma", "tD_2Dcutsigma", "_g0_", "intensities"').split(",")

        elif method in ["acf-1DCNN", "acf-LSTM"]:
                nn_params = ["ACF"]
            
        elif method in ["v-CNN", "i-CNNLSTM","i-3DCNN"]:
                nn_params = ["timeseries"]
        
        score = 0
        for el in nn_params: 
            #UPDATE
            if el not in ["_m_mono_", "t_mono", "_m1_", "_m2_", "t1", "t2", "N", "kappa", "tD", "_g0_", "R", "timeseries", "ACF"]:
                print("Choose a valid input parameter!")
                score += 1
        
        if score == 0: 
            break

    return nn_params

def fit_choice():
    fits = ["single", "double", "2DTIRF", "3DTIRF", "2Dcutsigma"]
    while True: 
        fit = input('Choose one of the following fits: "single", "double", "2DTIRF", "3DTIRF", "2Dcutsigma": ')

        if fit not in fits: 
            print("Choose a valid fit!")
        else: 
            break
    return fit


def set_choices_vars(method): 

    if method in ["p-CNN", "pf-LSTM"]:
        fit = fit_choice()

    elif method in ["acf-LSTM", "acf-1DCNN", "v-CNN", "i-CNNLSTM",  "i-3DCNN"]:
        fit = "none"

    nn_params = input_params(fit, method)

    
    return fit, nn_params


def set_dependent_vars(method, keyword):

    
    flatten = False
    split_ACF = False
    split_timeseries = False
    u_keyword = keyword

    if method == "p-CNN":
        model = "CNN"
    
    elif method == "pf-LSTM":
        model = "LSTM"
        flatten = True
        
    elif method == "acf-LSTM":
        model = "LSTM"
        flatten = True
        u_keyword = "20x20"
        
    elif method == "acf-1DCNN":
        model = "1DCNN"
        split_ACF = True
        u_keyword = "20x20"
        
    elif method == "v-CNN":
        #just to show that one SHOULD NOT use this method (see report)

        model = "CNN"
        split_timeseries = True
        u_keyword = "20x20"

    elif method == "i-CNNLSTM":
        model = "CNN_LSTM"
        u_keyword = "20x20"

    elif method == "i-3DCNN":
        #"3D block" of intensities fed in
        #could try 3D block of ACFS... but this method has not worked so far
        #NOT WORKING
        model = "3DCNN"
        u_keyword = "20x20"

    return model, flatten, split_ACF, split_timeseries, u_keyword


def initialise_variables():

    while True:
        s = input("Type D for default, C for choices: ")
        if s not in ["D", "C"]: 
            print("Not a valid option!")
        else: 
            break

    #s = "D"
    if s == "D": 
        antibiotic, method, fit, nn_params, keyword = def_antibiotic, def_method, def_fit, def_nn_params, def_keyword

    elif s == "C":

        antibiotic_options = ["Gentamycin", "Kanamycin", "Trimethoprim", "all"]
        while True: 
            antibiotic = input('Choose one of the following antibiotics: ', antibiotic_options)

            if antibiotic not in antibiotic_options: 
                print("Choose a valid antibiotic!")
            else: 
                break

        keywords = ["20x20", "BIN", "stdROI", "rot_300x4", "300x4"]
        while True: 
            keyword = input('Choose one of the following keywords: ', keywords )

            if keyword not in keywords: 
                print("Choose a valid keyword!")
            else: 
                break

        methods = ["p-CNN", "p-CNN-300x4", "pf-LSTM", "acf-1DCNN", "acf-LSTM", "v-CNN", "i-CNNLSTM"]
        while True: 
            method = input('Choose one of the following methods: ', methods )

            if method not in methods: 
                print("Choose a valid method!")
            else: 
                break

        fit, nn_params = set_choices_vars(method)
        

    model, flatten, split_ACF, split_timeseries, keyword = set_dependent_vars(method, keyword)

    if "300x4" in keyword:
        ROI = "300x4"
    elif keyword == "BIN":
        ROI = "60x60"
    else:
        ROI = "20x20"

    return Variables(antibiotic, keyword, method, fit, nn_params, model, flatten, split_ACF, split_timeseries, ROI)


variables = None

def get_variables(): 

    global variables
    if variables is None:
       variables = initialise_variables()
    return variables




