import torch
import torch.nn

from arg import BATCH_SIZE, TIMESTEPS, GRU_NAN


class GruCell(nn.Module):
    
    def __init__(self, channels_size):
        super(GruCell, self).__init__()
        self.conv_relu = nn.Sequential(nn.Conv2d(in_channels=channels_size, out_channels=channels_size,
        										 kernel_size=3, stride=1, padding=1),
                                       nn.ELU(),
                                       nn.Dropout(p=0.2))
        
        self.conv_relu_2x = nn.Sequential(nn.Conv2d(in_channels=channels_size+channels_size, out_channels=channels_size,
        										    kernel_size=3, stride=1, padding=1),
                                          nn.ELU(),
                                          nn.Dropout(p=0.2))
        
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

     
    def forward(self, x, hidden):
        input = torch.cat([x, hidden],dim=1)

        update_gate = self.conv_relu_2x(input)
        update_gate = self.sig((update_gate)) ### output after update gate
        reset_gate = self.conv_relu_2x(input)
        reset_gate = self.sig((reset_gate)) ### output after reset gate
        
        
        memory_gate_for_input = self.conv_relu(x)
        memory_gate_for_hidden = self.conv_relu(hidden)

        memory_content = memory_gate_for_input + (reset_gate * memory_gate_for_hidden) ### output for reset gate(affects how the reset gate do work)
        memory_content = self.relu(memory_content)

        hidden = (update_gate * hidden) + ((1 - update_gate) * memory_content) # torch.ones(input_size, hidden_size)

        return hidden, hidden


class Gru(nn.Module):

    def __init__(self, channels_size, gru_input_size, batch_size,
    			 timesteps, gru_nan): # arg for gru layer
        super(Gru, self).__init__()
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.channels_size = channels_size
        self.input_size = gru_input_size
        self.hidden_size = (self.batch_size, self.channels_size, self.input_size, self.input_size)
        
        self.gru_layer0 = GruCell(self.channels_size)
        self.init_hidden = torch.zeros(self.hidden_size).to(device)
        self.gru_nan = gru_nan


    def forward(self, x):
        x_cells = None
        x_list = []
        if self.gru_nan == False:
            try:
                x = x.reshape(self.batch_size, self.timesteps, self.channels_size, self.input_size, self.input_size)
                x = x.permute(1, 0, 2, 3, 4)
                for i in range(timesteps):
                    if x_cells is None:
                        x_cells, hidden = self.gru_layer0(x[i], self.init_hidden)
                        x_list.append(x_cells)
                    else:
                        x_i, hidden = self.gru_layer0(x[i], hidden)
                        x_list.append(x_i)
                x_cells = torch.stack(x_list)

             ##### FOR LAST BATCH
            except RuntimeError:
                x = x.reshape(1, self.timesteps, self.channels_size, self.input_size, self.input_size) #last batch is (15), but batch_size = 16, #arg.timesteps = 2 
                x = x.permute(1, 0, 2, 3, 4)
                hidden_zero = torch.zeros_like(x)
                for i in range(timesteps):
                    if x_cells is None:
                        x_cells, hidden = self.gru_layer0(x[i], hidden_zero[0])
                        x_list.append(x_cells)
                    else:
                        x_i, hidden = self.gru_layer0(x[i], hidden)
                        x_list.append(x_i)
                x_cells = torch.stack(x_list)
             #####
        elif self.gru_nan == True:
            try:
                x = x.reshape(self.batch_size, self.timesteps, self.channels_size, self.input_size, self.input_size)
                x = x.permute(1, 0, 2, 3, 4)
                for i in range(timesteps):
                    if x_cells is None:
                        x_cells, hidden = self.gru_layer0(x[i], self.init_hidden)
                        x_list.append(x_cells)
                    else:
                        x_i, hidden = self.gru_layer0(x[i], self.init_hidden)
                        x_list.append(x_i)
                x_cells = torch.stack((x_cells, x_i))
            ##### FOR LAST BATCH
            except RuntimeError:
                x = x.reshape(1, self.timesteps, self.channels_size, self.input_size, self.input_size) #last batch is (15), but batch_size = 16, #arg.timesteps = 2 
                x = x.permute(1, 0, 2, 3, 4)
                hidden_zero = torch.zeros_like(x)
                for i in range(timesteps):
                    if x_cells is None:
                        x_cells, hidden = self.gru_layer0(x[i], hidden_zero[0])
                        x_list.append(x_cells)
                    else:
                        x_i, hidden = self.gru_layer0(x[i], hidden)
                        x_list.append(x_i)
                x_cells = torch.stack((x_cells, x_i))
        else:
            print('gru_nan can be only True or False')
            quit()
        x_cells = x_cells.reshape(-1, self.channels_size, self.input_size, self.input_size)

        return x_cells  

