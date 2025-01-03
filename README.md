
## SCFIDNN: A deep Learning AST model for SCFI data

Classifies whether a strain (in this case E.coli) is resistant or susceptible to the antibiotics Kanamycin, Trimethorpim, and Gentamycin, based on data gathered from the Sub-Cellular Fluctuation Imaging technique developed at the University of Bristol. 

Pre-requesites:
- Folder with the saved arrays of the input parameters
- The experimental spreadsheets generated from running Arthur's code.

MORE DESCRIPTION (esp of Arthur's code)

### How to run

To (), run the "SCFIDNN_run_script.py" script. 

### Parameters

Before running the code, parameters need to be specified in the "set_variables.py" script.

General parameters are stored in the "Variables" class, while model hyperparameters are stored in the "Hyperparameters class.

Those variables within the classes that are set with arguments (?how do i say this?) can be modified directly within the class.
Running "SCFIDNN_run_script.py" will give the user two options on how to set the remaining variables: 
- One can either resort to default values, which are pre-emptively specified at the beginning of the "set_variables.py" script (those starting with "def_"), 
- or one can choose to be guided through the possible options via user input. 

If not familiar with the framework it is recommended to go through the input options, as a few methods are dependent on other specific setting, and all the possible options are also listed for an overview. 

#### Parameters: 

##### Data and labels

For details on the data used, see the "Data" subsection in the report. 

- gentamycin_folders, kanamycin_folders, trimethoprim_folders: 
    - each of these lists stores all Data-folder names for the corresponding antibiotic. Here, this includes 10 folder-names corresponding to the 10 E-coli strains treated with Gentamycin, and 3 folders for Kanamycin and Trimethoprim corresponding to the 3 repeats. 
- genta_y, kana_y, trime_y: The corresponding resistant (1) or susceptible (0) labels. 
    - For Gentamycin, the labelling is done based on MIC values: those strains with a MIC under 16 were are classified as susceptible 16 or over as resistant. 
    - For Kanamycin and Trimethoprim the data structure is more complicated: the data for "resistant" and "susceptible" are contained (stacked) each repeat folder. To handle this later, two folder names (for resistant and suceptible) are created for each repeat (for more details see the "pre_processing" function.)







