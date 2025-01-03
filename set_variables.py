
from initial_functions import res_sus_distinguisher,  set_choices_vars
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
def_model = "CNN_LSTM"
def_keyword = "BIN"
def_filter_keyword = False

def_flatten = False
def_split_ACF = False
def_split_timeseries = False


################################################################################



class Variables(): 
    def __init__(self, antibiotic, keyword, filter_keyword, method, fit, nn_params, model, flatten, split_ACF, split_timeseries, ROI):
        
        self.directory = "D:/Saved arrays"
        self.spreadsheets_dir = "D:\Data spreadsheets"

        self.antibiotics_dict = {
            "Gentamycin": {"folders": gentamycin_folders, "y_vals": genta_y},
            "Kanamycin": {"folders": kana_res_sus, "y_vals": kana_y},
            "Trimethoprim": {"folders": trime_res_sus, "y_vals": trime_y},
        }
        self.spec_thresh = 0.5
        self.use_thresh = False
        self.equal_Genta = True
        self.cross_val = False
        self.all_keywords = ["BIN", "stdROI", "rot_300x4", "300x4"]
        self.keyword = keyword
        self.filter_keyword = filter_keyword
        self.validation = True
        self.get_shap = True
        self.get_splits_i = True
        self.save_splits = False
        self.save_res = True
        self.log_tensorboard = False # IMPLEMENT

        self.antibiotic = antibiotic
        self.method = method
        self.fit = fit
        self.nn_params = nn_params
        self.model = model

        self.flatten = flatten
        self.split_ACF = split_ACF
        self.split_timeseries = split_timeseries
        self.ROI = ROI


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


def initialise_variables():

    while True:
        s = input("Type D for default, C for choices: ")
        if s not in ["D", "C"]: 
            print("Not a valid option!")
        else: 
            break

    #s = "D"
    if s == "D": 
        antibiotic, method, fit, nn_params, model, flatten, split_ACF, split_timeseries, filter_keyword= def_antibiotic, def_method, def_fit, def_nn_params, def_model, def_flatten, def_split_ACF, def_split_timeseries, def_filter_keyword

    elif s == "C":

        antibiotic_options = ["Gentamycin", "Kanamycin", "Trimethoprim", "all"]
        while True: 
            antibiotic = input('Choose one of the following antibiotics: "Gentamycin", "Kanamycin", "Trimethoprim", "all": ')

            if antibiotic not in antibiotic_options: 
                print("Choose a valid antibiotic!")
            else: 
                break


        methods = ["p-CNN", "p-CNN-300x4", "pf-LSTM", "acf-1DCNN", "acf-LSTM", "v-CNN", "i-CNNLSTM"]

        while True: 
            method = input('Choose one of the following methods: "p-CNN", "pf-LSTM", "acf-1DCNN", "acf-LSTM", "v-CNN", "i-CNNLSTM":  ' )

            if method not in methods: 
                print("Choose a valid method!")
            else: 
                break

        fit, nn_params, model, flatten, split_ACF, split_timeseries, filter_keyword = set_choices_vars(method, def_filter_keyword)


    if "300x4" in def_keyword and filter_keyword == True:
        ROI = "300x4"
    elif def_keyword == "BIN" and filter_keyword == True:
        ROI = "60x60"
    else:
        ROI = "20x20"

    return Variables(antibiotic, def_keyword, filter_keyword, method, fit, nn_params, model, flatten, split_ACF, split_timeseries, ROI)


variables = None

def get_variables(): 

    global variables
    if variables is None:
       variables = initialise_variables()
    return variables




