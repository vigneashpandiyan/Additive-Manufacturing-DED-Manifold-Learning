import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt # plotting library
import torch


class Encoder(nn.Module):    
    def __init__(self,dropout):
        super(Encoder, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(4),
                                     nn.ReLU(),
                                     
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(8),
                                     nn.ReLU(),
                                     
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3,stride=2),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                    )

        self.fc = nn.Sequential(nn.Linear(64 *14 * 9, 1024),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(1024,512),
                                
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(-1, 64 *14 * 9)
        output = self.fc(output)
        return output


class Decoder(nn.Module):
    
    def __init__(self, dropout):
        super(Decoder, self).__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(512, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, 64 *14 * 9),
            nn.ReLU(),
            
        )

        ### Unflatten
        self.unflatten = nn.Sequential(nn.Unflatten(dim=1, unflattened_size=(64, 14, 9)),
                                       #Print
                                       nn.BatchNorm2d(64))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            
            # First transposed convolution
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm2d(32),
            
            # Second transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm2d(16),
            
            # Third transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm2d(8),
            
            # fourth transposed convolution
            nn.ConvTranspose2d(8, 4, 3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm2d(4),
            
            # fivth transposed convolution
            nn.ConvTranspose2d(4, 3, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
        )
        
    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        return x
    
#%%

class Discriminator(nn.Module):    
    def __init__(self,dropout):
        super(Discriminator, self).__init__()
        self.convnet = nn.ModuleList([nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(4),
                                     nn.ReLU(),
                                     
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(8),
                                     nn.ReLU(),
                                     
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     #Print
                                     
                                     # nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3,stride=2),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     
                                     nn.Dropout(dropout),
                                     #Print
                                     ])

        self.fc = nn.ModuleList([
                                nn.Flatten(),
                                nn.Linear(8064, 512),
                                nn.ReLU(),
                                
                                nn.Dropout(dropout),
                                nn.Linear(512,64),
                                
                                nn.Dropout(dropout),
                                nn.Linear(64,1),
                                nn.BatchNorm1d(1),
                                nn.Sigmoid()]
                                )

    def forward(self, x):
        featurebank = []

        for idx_l, layer in enumerate(self.convnet):
            x = layer(x)
            
            if("torch.nn.modules.activation" in str(type(layer))):
                
                featurebank.append(x)
        convout = x
        

        for idx_l, layer in enumerate(self.fc):
            x = layer(x)
            
            if("torch.nn.modules.activation" in str(type(layer))):
                featurebank.append(x)
        disc_score = x
        

        return disc_score, featurebank

