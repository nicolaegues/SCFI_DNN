

def print_input_summary(antibiotic, method, nn_params, ROI, fit): 

    print("NN classes: 0-susceptible and 1-resistant")
    print("Antibiotic(s) considered: ", antibiotic)
    print("Method chosen: ", method)
    print("Params to be processed and inputted into NN: ", nn_params)
    print("ROI size:", ROI)
    print("Fit used: ", fit)

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
            if el not in ["_m_mono_", "t_mono", "_m1_", "_m_2_", "t1", "t_2", "N", "kappa", "tD", "_g0_", "R", "timeseries", "ACF"]:
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


def set_choices_vars(method, filter_keyword): 

    flatten = False
    split_ACF = False
    split_timeseries = False
    u_filter_keyword = filter_keyword



    if method == "p-CNN":
        fit = fit_choice()
        nn_params = input_params(fit, method)
        model = "CNN"
    
    elif method == "pf-LSTM":
        #flattened fit params 
        fit = fit_choice()
        nn_params = input_params(fit, method)
        model = "LSTM"
        flatten = True
        

    elif method == "acf-LSTM":
        #flattened ACFs 
        fit = "none"
        nn_params = input_params(fit, method)
        model = "LSTM"
        flatten = True
        u_filter_keyword = False
        
    elif method == "acf-1DCNN":
        fit = "none"
        nn_params = input_params(fit, method)
        model = "1DCNN"
        split_ACF = True
        u_filter_keyword = False
        
    elif method == "v-CNN":
        #just to show that one SHOULD NOT use this method (see report)
        fit = "none"
        nn_params = input_params(fit, method)
        model = "CNN"
        split_timeseries = True
        u_filter_keyword = False

    elif method == "i-CNNLSTM":
        fit = "none"
        nn_params = input_params(fit, method)
        model = "CNN_LSTM"
        u_filter_keyword = False

    elif method == "i-3DCNN":
        #"3D block" of intensities fed in
        #could try 3D block of ACFS... but this method has not worked so far
        #NOT WORKING
        fit = "none"
        nn_params = input_params(fit, method)
        model = "3DCNN"
        u_filter_keyword = False
    
    return fit, nn_params, model, flatten, split_ACF, split_timeseries, u_filter_keyword



def res_sus_distinguisher(ab_names):
        
    new_names = []
    yvals = []
    for name in ab_names: 

        new_names.append(name + "_r")
        yvals.append(1)

        new_names.append(name + "_s")
        yvals.append(0)
    
    return new_names, yvals
