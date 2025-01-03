
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

If not familiar with the framework it is recommended to go through the input options, as a few methods are dependent on other specific settings (and all the possible options are also listed).

Parameters: 





