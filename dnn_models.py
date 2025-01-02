#set up of CNN model structure

#better way to determine the sizes of things


import torch
from torch import nn


def calculate_flat_dim(im_size, hidden_dim, nr_layers, pool_kernel_size):

    height = im_size[0]
    width = im_size[1]

    for layer in range(nr_layers):
        height //= pool_kernel_size[0]
        width //= pool_kernel_size[1]

    final_hidden_dim = hidden_dim*(2**(nr_layers-1))
    flat_dim = height * width * final_hidden_dim

    return flat_dim

test_2 = calculate_flat_dim((20, 20), 32, 4, (2, 2))
test_3 = calculate_flat_dim((4, 300), 32, 2, (1, 4))
pass

class CNN_Model(nn.Module):
    def __init__(self, nr_params, keyword):
        super(CNN_Model, self).__init__()

        input_dim = nr_params
        hidden_dim = 32


        if "300x4" not in keyword: 
            kernel_size = 3
            pool_kernel_size = 2

            if keyword == "BIN":
                flat_dim = 2304 #60x60
            else: 
                flat_dim = 256 #20x20

        else: 
            kernel_size=(2, 4)
            pool_kernel_size = (1, 4)
            flat_dim = 4800

        flat_dim = calculate_flat_dim(nr_params, )
   
       
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, padding="same"),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pool_kernel_size))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=kernel_size, padding="same"),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pool_kernel_size))
        
        self.layer3= nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=kernel_size, padding="same"),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pool_kernel_size))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=kernel_size, padding="same"),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = pool_kernel_size))
        

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_dim, flat_dim //2),
            nn.ReLU())
    
        self.fc3= nn.Sequential(
            nn.Linear(flat_dim //2, 2))

        
    def forward(self, x, keyword, model):
        
        x = x.to(torch.float32)

        out = self.layer1(x)
        out = self.layer2(out)
    
        if "300x4" not in keyword: 
            out = self.layer3(out)
            out = self.layer4(out)

        out = torch.flatten(out, start_dim = 1)
        #flat_shape = 
        out = self.fc(out)

        #if model is CNN_LSTM then we instead want the feature vector outputted by the previous step.
        if model not in ["CNN_LSTM"]: 
            out = self.fc3(out)

        return out

class LSTM_Model(nn.Module):
    def __init__(self, seq_length):
        super(LSTM_Model, self).__init__()
    
        input_dim=seq_length
        hidden_dim = 64

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first= True)
        self.sequential = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(hidden_dim, 2)
        

    def forward(self, x):

        x = x.to(torch.float32)
        x, _ = self.lstm(x)
        x = self.sequential(x)
        x = x[:, -1, :]
        x = self.fc(x)
      
        return x
    
class GRU_Model(nn.Module):
    def __init__(self, seq_length):
        super(GRU_Model, self).__init__()

        input_dim=seq_length
        hidden_dim = 64


        self.rnn1 = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=0)
        self.rnn2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)


    def forward(self, x):
        x = x.to(torch.float32)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        # Extract the last time step output
        x = x[:, -1, :]
        x = nn.functional.dropout(x, p=0.4)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x

class C3D_Model(nn.Module): 
    def __init__(self, nr_params):

        super(C3D_Model, self).__init__()
        input_dim = nr_params
        hidden_dim = 32
        flat_dim = 4608

        self.layer1 = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, kernel_size=3, padding=1),
            #nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.layer2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            #nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.layer3 = nn.Sequential(
            #nn.Conv3d(128, 256, kernel_size=3, padding=1),
            #nn.BatchNorm3d(256),
            #nn.ReLU(),
            nn.Conv3d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1),
            #nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.layer4 = nn.Sequential(
            #nn.Conv3d(256, 512, kernel_size=3, padding=1),
            #nn.BatchNorm3d(512),
            #nn.ReLU(),
            nn.Conv3d(hidden_dim*4, hidden_dim*8, kernel_size=3, padding=1),
            #nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        #last_duration = int(math.floor(sample_duration / 16))
        #last_size = int(math.ceil(sample_size / 32))


        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            #nn.Linear(12800, 4096),
            nn.Linear(flat_dim, flat_dim//2),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(flat_dim//2, 2))         

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, start_dim = 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

class C1D_Model(nn.Module):

    def __init__(self):

        super(C1D_Model, self).__init__()

        input_dim = 1
        hidden_dim = 32
        flat_dim = 3200

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim=32, kernel_size=(3,), padding="same"),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=(1,), padding="same"),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = (2,)))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flat_dim, flat_dim//2),
            nn.ReLU())
        
        self.fc2 = nn.Sequential(
            nn.Linear(flat_dim//2, 2)) 
        
    def forward(self, x):
        x = x.to(torch.float32)
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, start_dim = 1)
        #out = self.fc(out)
        out = self.fc2(out)
        return out 
    


temporal_filter_size = 25
momentum = 0.9
input_dim = 1
hidden_dim = 32

class Shortcut(nn.Module):

    def __init__(self):
        super(Shortcut, self).__init__()
        
    def forward(self, this_input, residual):
        # Truncate the end section of the input tensor
        shortcut = this_input[:, :, :-(2 * temporal_filter_size -2)]

        return shortcut + residual
    
class fcsnet(nn.Module):
    def __init__(self):
        super(fcsnet, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv1d(input_dim, hidden_dim, kernel_size=temporal_filter_size), 
                    #nn.BatchNorm1d(hidden_dim, momentum=momentum), 
                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=temporal_filter_size), 
                    #nn.BatchNorm1d(hidden_dim, momentum=momentum), 
                    nn.ReLU())
        self.shortcut = Shortcut()

        self.layer3 = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=temporal_filter_size), 
                    #nn.BatchNorm1d(hidden_dim, momentum=momentum), 
                    nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=temporal_filter_size), 
                    #nn.BatchNorm1d(hidden_dim, momentum=momentum), 
                    nn.ReLU())
        self.shortcut2 = Shortcut()


        #1x1 layers - 12 of them
        self.consecutive_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=(1,)), nn.ReLU()) for _ in range(12)]
        )


        self.fc = nn.Sequential(
                    nn.Linear(128, 2))
                

    def forward(self, x):
        x = x.to(torch.float32)
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.shortcut(this_input = identity, residual = out)

        identity = out
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.shortcut2(this_input = identity, residual = out)


        for i in range(0, len(self.consecutive_layers), 2):
            identity = out
            out = self.consecutive_layers[i](out)  #first layer
            out = self.consecutive_layers[i + 1](out)  #second layer
            out = identity + out

        out = torch.flatten(out, start_dim = 1)
        out = self.fc(out)
        
        return out