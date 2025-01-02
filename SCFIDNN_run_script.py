
"""
Classifies whether a strain (in this case E.coli) is resistant (1) or susceptible (0) when treated with Kanamycin, Trimethorpim, and Gentamycin, 
based on data gathered from the Sub-Cellular fluctuation technique developed at the University of Bristol -  video of the scattered light signals.

NEEDED for the code to run: Folder with the saved arrays of the input parameters, as well as the experimental spreadsheets generated from running Arthur's code.

When you run this code you're presented with various optons - choice of antibiotic, DNN method, choice of fit and  choice of fit paramter to use as input. 
See lowest part of code for individual descriptions. 


########################################################################
#Output classes
#-----------------------------------------------------------------------
#0 - susceptible, 1 - resistant



########################################################################
#nnparams: Parameter input choice
#-----------------------------------------------------------------------
#"_m_mono_", "t_mono" belong to the single exponential fit. 
#"_m1_", "_m_2_", "t1", "t_2" belong to the double exponential fit. 
#"N", "kappa", "tD" belong to the TIRF fit (3D or 2D)
#"_g0_", "intensities": fit-independent
#"R": Rsquared
#"timeseries": intensity timeseries
#"ACF": Autocorrelation function


########################################################################
#Fit choice 
#-----------------------------------------------------------------------
#"single": single exponential fit
#"double": double exponential fit
#"2DTIRF": 2D TIRF (with G_inf)
#"3DTIRF": 3D TIRF (no G_inf)
#"2Dcutsigma": 2D TIRF whereby first point of ACFs were cut off and fit was forced through the second point (first point in new ACF)
#"none": when either ACFs or timeseries wants to be inputted. Mostly because these were only saved for one fit.

#-----------------------------------------------------------------------

########################################################################
#Method choice
#p-CNN: input is fit params
#v-CNN: input is individual video frames - should not be used, except if for later cnn
#acf-1DCNN: input: individual ACF curves (no vid distinction)
#acf-LSTM: input: flattened ACFs for each vid
#i-CNNLSTM: input is intensity timeseries. Requires having trained a CNN on the antibiotics individual frames beforehand, to then load in its state dict for feature extraction
#i-3DCNN: input is intensity timeseries. DOES NOT WORK
#pf-LSTM: flattened params for each vid. DOES NOT WORK

"""


"""

TO-DO's: 

- for non cross-val: do such that each run a different separate 2/3  is used as training
        - get specific repeat separators. 
        [r, s,| r, s, | r, s] (trime, kana)
        

- build in sth to automatically do shap plot differently for 300x4

- (tensorflow option)

"""




from pre_processing import param_data_processing
from dnn_run_funcs import run_CNN_LSTM_models
from  set_variables import get_variables
from initial_functions import print_input_summary



v = get_variables()
cross_val = v.cross_val
antibiotic = v.antibiotic

#filter keyword and keyword, and model
print_input_summary(antibiotic, v.method, v.nn_params, v.ROI, v.fit)


total_X_arr, total_Y_arr, G_total_X_tr_arr, G_total_X_ts_arr, G_total_Y_tr_arr, G_total_Y_ts_arr  = param_data_processing( )


run_CNN_LSTM_models(total_X_arr, total_Y_arr, G_total_X_tr_arr, G_total_X_ts_arr, G_total_Y_tr_arr, G_total_Y_ts_arr)

#modeltoconfig()
pass


