

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
from helper_functions import print_input_summary



v = get_variables()
cross_val = v.cross_val
antibiotic = v.antibiotic

#filter keyword and keyword, and model
print_input_summary(antibiotic, v.method, v.nn_params, v.ROI, v.fit)


total_X_arr, total_Y_arr, G_total_X_tr_arr, G_total_X_ts_arr, G_total_Y_tr_arr, G_total_Y_ts_arr  = param_data_processing( )


run_CNN_LSTM_models(total_X_arr, total_Y_arr, G_total_X_tr_arr, G_total_X_ts_arr, G_total_Y_tr_arr, G_total_Y_ts_arr)

#modeltoconfig()
pass


