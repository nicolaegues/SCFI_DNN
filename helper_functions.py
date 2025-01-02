from torch.utils.data import Dataset, DataLoader
from torchinfo import summary  
from sklearn.model_selection import train_test_split, StratifiedKFold
from dnn_models import CNN_Model, LSTM_Model, GRU_Model, C3D_Model, C1D_Model, fcsnet


class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_x = self.X[idx]
        sample_y = self.Y[idx]

        # Apply transformation if provided
        if self.transform:
            sample_x = self.transform(sample_x)
        
        return sample_x, sample_y
    
def reset_model(chosen_model):
    for layer in chosen_model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

def initialise(model, nr_params, seq_length, keyword):

    if model == "3DCNN": 
        my_model = C3D_Model(nr_params)

    elif model in ["CNN_LSTM","LSTM"]:
        #my_model = GRU_Model()
        my_model = LSTM_Model(seq_length)
        #batch_size = 16
        batch_size = 64

    elif model == "1DCNN": 
        #my_model = C1D_Model()
        my_model = fcsnet()
        batch_size = 5000

    else:
        my_model = CNN_Model(nr_params, keyword)
        #batch_size = 32
        batch_size = 32   

    #just in case? tho shouldn't be necessary
    my_model.apply(reset_model)

    return my_model, batch_size

def get_summary(my_model, dataset, batch_size): 

    dataloader = DataLoader(dataset, shuffle = True, batch_size=batch_size)


    #look at shape of typical batches in data loaders
    for idx, (X_, Y_) in enumerate(dataloader):
        print("X: ", X_.shape)
        print("Y: ", Y_.shape)
        if idx >= 0:
            break

    #model architecture summary
    summary(my_model,
            input_data = X_,
            col_names=["input_size",
                        "output_size",
                        "num_params"])

def get_val_splits(trainval_idx, dataset): 
    #splits the 4/5 of original data further into training and validation
    #such that in total, 3/5 of folds are used for training, 1/5 for validation and 1/5 for testing
    #(and splits are different for each fold)

    trainval_dataset = CustomDataset(dataset[trainval_idx][0], dataset[trainval_idx][1])
    

    kfold2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, val_idx = next(kfold2.split(trainval_dataset, trainval_dataset.Y))

    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]


    return train_idx, val_idx

def get_non_sampling_loaders(X_trainval, Y_trainval, X_test, Y_test, batch_size):
    #NOTE this train/val/test splitting is DIFFERENT than the fold method: 
    #Here an extra measure of generalizability: Data is split into train/val set and testset NOT by random splitting. (shuffling does occur though after the splitting) 
    #This splitting only applicable to KANAMYCIN and TRIMETHOPRIM given that they have three repeats and so we know that the last repeat has an equal nr of classe


    testset = CustomDataset(X_test, Y_test)
    trainvalset = CustomDataset(X_trainval, Y_trainval)
    
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, stratify=Y_trainval)
    trainset = CustomDataset(X_train, Y_train)
 
    validset = CustomDataset(X_val, Y_val)
  
    trainvalloader = DataLoader(trainvalset, shuffle = True, batch_size=batch_size)
    trainloader = DataLoader(trainset, shuffle = True, batch_size=batch_size)
    validloader = DataLoader(validset, shuffle = True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle = True, batch_size=batch_size)

    return trainloader, validloader, testloader, trainvalloader

