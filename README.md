
# SCFIDNN: A deep Learning AST model for SCFI data

Classifies whether a strain (in this case E.coli) is resistant or susceptible to the antibiotics Kanamycin, Trimethorpim, and Gentamycin, based on data gathered from the Sub-Cellular Fluctuation Imaging technique developed at the University of Bristol. 

Pre-requesites:
- Folder with the saved arrays of the input parameters
- The experimental spreadsheets generated from running Arthur's code.

MORE DESCRIPTION (esp of Arthur's code)

## How to run

To (), run the "SCFIDNN_run_script.py" script. 

## 1. Setting of variables
Before running the code, parameters need to be specified in the "set_variables.py" script.

### Parameters: 

#### Data and labels

For details on the data used for the Autocorrelation analysis, see the "Data" subsection in the report. 

Agh i actually need to talk abt saved_arrays folder though, not data folder

- gentamycin_folders, kanamycin_folders, trimethoprim_folders: 
    - Each of these lists stores all Data-folder names for the corresponding antibiotic. Here, this includes 10 folder-names corresponding to the 10 E-coli strains treated with Gentamycin, and 3 folders for Kanamycin and Trimethoprim corresponding to the 3 repeats. 
- genta_y, kana_y, trime_y: The corresponding resistant (1) or susceptible (0) labels. 
    - For Gentamycin, the labelling is done based on MIC values: those strains with a MIC under 16 were are classified as susceptible 16 or over as resistant. 
    - For Kanamycin and Trimethoprim the data structure is more complicated: the data for "resistant" and "susceptible" are contained (stacked) each repeat folder. To handle this later, two folder names (for resistant and suceptible) are created for each repeat (for more details see the "pre_processing" function.)

The folders and corresponding labels for each antibiotic are then stored in a nested dictionary within the "Variables" class.

- data_directory: 
- spreadsheets_dir: 


#### Non-permanent parameters 

The following five parameters can either be set to default values (specified at the beginning of the "set_variables.py" script), or set while running via user-input choices. 
If not familiar with the framework it is recommended to go through the input options, as a few methods are dependent on other specific setting, and all the possible options are also listed for an overview. 

- antibiotic: name of the antibiotic for which the DNN frameworks wants to be run. Via the data dictionary above this can then be used to access the corresponding data and labels. 

- keyword: ()
    - options are  "20x20", "BIN", "stdROI", "rot_300x4", "300x4"

- method: ()
    - "p-CNN": input is fit params
    - "v-CNN": input is individual video frames - should not be used, except if for later cnn
    - "acf-1DCNN": input: individual ACF curves (no vid distinction)
    - "acf-LSTM": input: flattened ACFs for each vid
    - "i-CNNLSTM": input is intensity timeseries. Requires having trained a CNN on the antibiotics individual frames beforehand, to then load in its state dict for feature extraction
    - "i-3DCNN": input is intensity timeseries. DOES NOT WORK
    - "pf-LSTM": flattened params for each vid. DOES NOT WORK

- fit: ()
    - "single": single exponential fit
    - "double": double exponential fit
    - "2DTIRF": 2D TIRF (with G_inf)
    - "3DTIRF": 3D TIRF (no G_inf)
    - "2Dcutsigma": 2D TIRF whereby first point of ACFs were cut off and fit was forced through the second point (first point in new ACF)
    - "none": when either ACFs or timeseries wants to be inputted. Mostly because these were only saved for one fit.

- nn_params: ()
    - "m_mono", "t_mono" belong to the single exponential fit. 
    - "m1", "m2", "t1", "t2" belong to the double exponential fit. 
    - "N", "kappa", "tD" belong to the TIRF fit (3D or 2D)
    - "g0", "intensities": fit-independent
    - "R": Rsquared
    - "timeseries": intensity timeseries
    - "ACF": Autocorrelation function

Method-dependant variables:

- model, flatten, split_ACF, split_timeseries

Given that various methods use the same DNN model, an additional variable specifying the model is set based on the method. This is for ease of use later.

The latter three parameters are boolean variables used in the pre_processing function to get the data into the correct format, depending on which DNN method is chosen. More info in the pre_processing section. 

#### More permanent parameters

()

These are set directly either in the "Variables" or the "Hyperparameters" classes. 

- General:

    - self.spec_thresh = 0.5
    - self.use_thresh = False
    - self.equal_Genta = True
    - self.cross_val = False
    - self.validation = True
    - self.get_shap = True
    - self.get_splits_i = True
    - self.save_splits = False
    - self.save_res = True


- hyperparams:

    - self.lstm_A_lr=0.005
    - self.lstm_early_stop_thresh = 10
    - self.lstm_max_epochs = 100
    - self.cnn_A_lr = 0.0005
    - self.cnn_early_stop_thresh = 10#5normal
    - self.cnn_max_epochs = 50 #30 normal
    - self.optimizer = torch.optim.Adam()
    - self.loss_fn = torch.nn.CrossEntropyLoss()
    - self.k_folds = 5

## 2. Pre-processing

## 3. Running the DNN







