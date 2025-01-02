
import re
import os
import sys
import numpy as np
import pandas as pd
from set_variables import get_variables

#why does id cross_val have to be false

v = get_variables()

directory = v.directory

antibiotic = v.antibiotic
antibiotics_dict = v.antibiotics_dict
nn_params = v.nn_params
keyword = v.keyword
all_keywords = v.all_keywords
equal_Genta = v.equal_Genta
filter_keyword = v.filter_keyword
split_timeseries = v.split_timeseries
split_ACF = v.split_ACF
flatten = v.flatten
cross_val = v.cross_val
spreadsheets_dir = v.spreadsheets_dir



########################################################################
#Pre-processing
#-----------------------------------------------------------------------
#Needed: the Saved Array folders and the Excel spreadsheets generated after Arthur's fitting code is run
########################################################################

def param_data_processing(save = "no"): 

    #intensities added seperately to make sure it's loaded in cases where it's needed for normalisation, even if don't want it as NN input
    #will be removed so that not inputted into NN - if want as input, specify in nn_params. 
    to_be_normalised_list = ["_m_mono_", "_m1_", "_m2_", "_g0_", "_g1_", "g0bifit", "g1bifit", "g0fast", "g1fast", "g0slow", "g1slow"]
    if "intensities" not in nn_params and any(substring in nn_params for substring in to_be_normalised_list):
        load_intensities = True
        params = nn_params + ["intensities"]

    else: 
        load_intensities = False
        params = nn_params

    #will normalise g0 or the m's with intensity**power
    power = 2


    total_X_list = []
    total_Y_list = []

    #this is for the case where we don't want to cross validate later but want to train on subsection of Gentamycin (the last repeat in each folder)    
    #and then test on the first two repeats in each folder
    total_X_train_list = []
    total_X_test_list = []
    total_Y_train_list = []
    total_Y_test_list = []

    
    #Main processing loop

    main_dict = antibiotics_dict[antibiotic]
    ab_folders = main_dict["folders"]
    ab_y_vals = main_dict["y_vals"]
    

    if equal_Genta == True and antibiotic == "Gentamycin": 
        ab_folders = [0, 1,  4,  7]

    for s in range(len(ab_folders)):

        print("Doing preprocessing for: ", ab_folders[s])
        
        #gets the Saved arrays folder name. for cases where "_r" or "_s" si in the name, it removes this 
        #to get the original folder name - whether there's an _r or _s will be important for later when only 
        #either the resistant or the susceptible files are kept
        if re.search(r'_r|_s', ab_folders[s]):
            folder = ab_folders[s][:-2]
        else: 
            folder = ab_folders[s]

        strain_arrays_dir = os.path.join(directory, folder)
        path, dirs, files = next(os.walk(strain_arrays_dir))

        row_pattern = re.compile("row_")
        files = [file for file in files if (row_pattern.search(file))]

        #sorts files by row nr
        p1 = re.compile(r'row_(\d+)')
        sorted_files = sorted(files, key=lambda s: int(p1.search(s).groups()[0]))

        keyword_pattern = re.compile(keyword)
        all_keyword_pattern = re.compile("|".join(re.escape( k )for k in all_keywords))
        rot_pattern =  re.compile("rot_300x4")

        if filter_keyword == True: 

            if keyword == "300x4": 
                 sorted_files = [file for file in sorted_files if (keyword_pattern.search(file)) and not (rot_pattern.search(file))]
            else:

                sorted_files = [file for file in sorted_files if (keyword_pattern.search(file))]

        
        #this if for the normal 20x20 case (where this is no keyword present)
        else: 
            #THIS COULD BE WRONG - CHECK!
            sorted_files = [file for file in sorted_files if not (all_keyword_pattern.search(file)) ]

        #only keeps files of selected parameters
        #fit_files = []
        pattern = re.compile("|".join(re.escape( p )for p in params))

        if "R" not in params: 
            #this is necessary because "R_tD" (saved R_squared file name for some) would be mistaken for tD by the pattern finder

            R_pattern = re.compile(r'_R_')
            fit_files = [file for file in sorted_files if (pattern.search(file) and not (R_pattern.search(file)))]

        else: 
            fit_files = [file for file in sorted_files if (pattern.search(file))]

        #to check for errors
        t = "".join(fit_files[:len(params)])
        if s == 0:
            print("Selected files of type: ", fit_files[:(len(params)*2)])

        for el in params: 
            if el not in t:
                sys.exit("Parameter files were not selected - probably because they're not in your folder!")
        
        #gets paths for the parameter files
        ordered_array_paths = []
        for el in fit_files:
            ordered_array_paths.append(os.path.join(strain_arrays_dir, el))

        #creates sublists for every param within the total list
        row_list = [ordered_array_paths[i:i+len(params)] for i in range(0, len(ordered_array_paths), len(params))]

        #finds out how many rows there are for each type (T/UT (Gentamycin), tr/ts/ur/ur(kanamycin, trimethoprim)), via the Excel spreadsheets for that saved array folder
        sheet_dir = os.path.join(spreadsheets_dir, folder +  ".xlsx")
        df = pd.read_excel(sheet_dir)
        col_vals = df["dividers"]
        col_vals = col_vals[col_vals != 0]
        div_list = col_vals.tolist()


        #loads the files and creates array with all data
        loaded_arrays = []
        for row in row_list: 

            loaded_row = [np.load(file_path) for file_path in row]
            loaded_arrays.append(loaded_row)

            
            if "timeseries" in nn_params:
                #some videos have different lengths, especially different for Gentamycin. This gets a uniform length for them.
                if antibiotic == "Gentamycin":
                    loaded_row[0] = loaded_row[0][:150]
                    l = loaded_row[0]
                else:
                    diff = 401- len(loaded_row[0])
                    if diff != 0:  
                        last_el = loaded_row[0][-1]
                        new_row = np.expand_dims(last_el, axis=0)
                        for i in range(diff): 
                            loaded_row[0] = np.concatenate((loaded_row[0], new_row), axis=0)
                        l = loaded_row[0]
            


        arr_loaded_arrays = np.array(loaded_arrays)
        #t = arr_loaded_arrays[0]
        
        #Divides any g0 or m by intensity-squared

        #print(arr_loaded_arrays[0][1][0])
        if load_intensities == True:

            print("Normalising with intensity...")

            #gets index of intensity column in array by seeing at which index intensity first appears in fit files
            for el in fit_files: 
                if "intensities" in el: 
                    i_index = fit_files.index(el)
                    break

            i_squared = (arr_loaded_arrays[0:, i_index])**power

            for sp in params: 
                #indexing is not gonna work if m1 and m2 in params, need to adapt later
                if sp in to_be_normalised_list:
                    for el in fit_files: 
                        if sp in el: 
                            index = fit_files.index(el)
                            break
                    normed_p = arr_loaded_arrays[0:, index]/i_squared
                    arr_loaded_arrays[0:, index] = normed_p

        

        #print(arr_loaded_arrays[0][1][0])

        """

        test if array transformations/calculations correct. if a/b and c are the same then yes. 
        a = arr_loaded_arrays[0][1]
        b = i_squared[0]
        c = normed_m[0][0]
        print(a/b)
        print(c) 
        """

        #normalise treated with untreated
        """
        def produce_norm_val(arrays):
        """
        """

            Takes an ndarray as input. 
            Returns the average value of every column of the array (every parameter).
            For every column (every parameter), the function calculates the median of the array in every row 
            and then takes the mean of these medians, giving one value per column (parameter).
        """
        """

            #prob need to change this for ACF

            #rows = np.array(rows)
            results = {}
            for p in range(len(sorted_params)):
                parameter = sorted_params[p] 
                results[parameter] = [] 
                avgs = []
                for row in range(len(arrays)):
                    #print(rows[row number, column number])
                    arr = arrays[row, p]
                    avgs.append(np.median(arr.flatten()))
                results[parameter].append(np.mean(avgs))
            return results
        
        
        norm_vals = produce_norm_val(untreated_array)
        print(norm_vals)

       
        #divide each param in treated array by average of untreated equivalent.
        for i in range(len(norm_vals)):
            p_val = list(norm_vals.values())[i]
            treated_array[0:, i] = treated_array[0:, i]/p_val
        """
        
            
        #remove intensity from treated array in case it's not wanted as an input to the NN.
        treated_array = arr_loaded_arrays

        if antibiotic == "Gentamycin":
            treated_array = arr_loaded_arrays[:div_list[2]] 
            untreated_array = arr_loaded_arrays[div_list[2]: div_list[-1]] 
      
        elif re.search(r'_r', ab_folders[s]):
            treated_array = arr_loaded_arrays[:div_list[0]]
            untreated_array = arr_loaded_arrays[div_list[1]:div_list[2]]

        elif re.search(r'_s', ab_folders[s]):
            treated_array = arr_loaded_arrays[div_list[0]:div_list[1]]
            untreated_array =arr_loaded_arrays[div_list[2]: div_list[-1]]

        if load_intensities == True: 
            treated_array = np.delete(treated_array, i_index, axis=1)


        #X_array = np.log(treated_array[:len(untreated_array)] /untreated_array[:len(treated_array)])
        X_array = treated_array
        Y_array = [ab_y_vals[s]]*len(X_array)


        if "ACF" in nn_params or "timeseries"  in nn_params: 
    
            if split_timeseries == True:
                #frame by frame split
                X_array = X_array.transpose(0, 2, 3, 4, 1)

                frames = []
                for vid in X_array:
                    for frame in vid:
                        frames.append(frame)
                X_array = np.array(frames)
                X_array = X_array.transpose(0, 3, 2, 1)
                Y_array = [ab_y_vals[s]]*len(X_array)


            if split_ACF == True: 
                #this is for the Conv1D
                #gets the ACF for every pixel individually
                X_array = X_array.transpose(0, 2, 3, 4, 1)

                ACF_curves = []
                for vid in range(len(X_array)):
                    for i in range(20):
                        for j in range(20):
                            ACF_curve = X_array[vid, :, i, j]
                            ACF_curves.append(ACF_curve)

                ACF_curves = np.array(ACF_curves)
                
                #tests to see whether ACF curves array is correct
                """
                xaxis = np.linspace(0, 100, 100)
                for i in range(len(ACF_curves)):
                    y = X_array[i]
                    plt.figure()
                    plt.scatter(xaxis, y)
                    plt.show()
                """

                #mean = np.mean(ACF_curves, axis=1, keepdims = True)
              
                #needs to be transposed this way for cnn input
                ACF_curves = ACF_curves.transpose(0, 2, 1)
                X_array = ACF_curves
                Y_array = [ab_y_vals[s]]*len(X_array)  


        if flatten == True: 
            
            flattened = []
            for vid in X_array: 
                flattened.append(vid.flatten())

            X_array = np.array(flattened)
            #X_array = X_array.flatten()
            Y_array = [ab_y_vals[s]]*len(X_array)
            X_array = np.expand_dims(X_array, 1)
            

        total_X_list.append(X_array)
        total_Y_list.extend(Y_array)

        if cross_val == False and antibiotic == "Gentamycin": 
            total_X_train_list.append(X_array[:int(len(X_array)*2/3)])
            total_X_test_list.append(X_array[-int(len(X_array)*1/3):])

            total_Y_train_list.append(Y_array[:int(len(Y_array)*2/3)])
            total_Y_test_list.append(Y_array[-int(len(Y_array)*1/3):])

    separators = [len(sublist) for sublist in total_X_list]
    total_X_arr = np.concatenate(total_X_list)
    total_Y_arr = np.array(total_Y_list)
    

    if cross_val == False and antibiotic == "Gentamycin": 
        G_total_X_tr_arr = np.concatenate(total_X_train_list)
        G_total_X_ts_arr = np.concatenate(total_X_test_list)

        G_total_Y_tr_arr = np.concatenate(total_Y_train_list)
        G_total_Y_ts_arr = np.concatenate(total_Y_test_list)

    if save == "yes": 
        if "timeseries" in nn_params: 
            np.save("D:/train_test arrays/"+antibiotic +"/Xarr_" + antibiotic + "_" + str(nn_params) + "_split_" + str(split_timeseries), total_X_arr)
            np.save("D:/train_test arrays/"+antibiotic +"/Yarr_" + antibiotic + "_" + str(nn_params) + "_split_" + str(split_timeseries), total_Y_arr)
        else: 
            np.save("D:/train_test arrays/"+antibiotic +"/Xarr_" + antibiotic + "_" + str(nn_params) + "_300x4"  , total_X_arr)
            np.save("D:/train_test arrays/"+antibiotic +"/Yarr_" + antibiotic + "_" + str(nn_params) + "_300x4", total_Y_arr)

    if cross_val == False and antibiotic == "Gentamycin": 
        total_X_arr, total_Y_arr = None, None
    
    else:
        G_total_X_tr_arr, G_total_X_ts_arr, G_total_Y_tr_arr, G_total_Y_ts_arr  = None, None, None, None
    
    return total_X_arr, total_Y_arr, G_total_X_tr_arr, G_total_X_ts_arr, G_total_Y_tr_arr, G_total_Y_ts_arr 
    
