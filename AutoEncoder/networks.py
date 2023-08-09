import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt # plotting library
import torch

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
        
    
    def forward(self, x):
        # Do your print / debug stuff here
        # print('Layer No', Layerno)
        # print('Encoder Dimension ',x.shape)
        return x
    
class PrintLayer_1(nn.Module):
    def __init__(self):
        super(PrintLayer_1, self).__init__()
        
    
    def forward(self, x):
        # Do your print / debug stuff here
        # print('Layer No', Layerno)
        # print('Decoder Dimension ',x.shape)
        return x


class Encoder(nn.Module):    
    def __init__(self,dropout):
        super(Encoder, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(4),
                                     nn.ReLU(),
                                     #Print
                                     PrintLayer(),
                                     # nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(8),
                                     nn.ReLU(),
                                     #Print
                                     PrintLayer(),
                                     # nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     #Print
                                     PrintLayer(),
                                     # nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3,stride=2),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     #Print
                                     PrintLayer(),
                                     # nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=2),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     #Print
                                     PrintLayer(),
                                     # nn.MaxPool2d(2,2),
                                     nn.Dropout(dropout),
                                     #Print
                                     PrintLayer())

        self.fc = nn.Sequential(nn.Linear(64 *14 * 9, 1024),
                                nn.ReLU(),
                                #Print
                                PrintLayer(),
                                nn.Dropout(dropout),
                                nn.Linear(1024,512),
                                #Print
                                PrintLayer()
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
            PrintLayer_1(),
            # Second linear layer
            nn.Linear(1024, 64 *14 * 9),
            nn.ReLU(),
            #Print
            PrintLayer_1()
        )

        ### Unflatten
        self.unflatten = nn.Sequential(nn.Unflatten(dim=1, unflattened_size=(64, 14, 9)),
                                       #Print
                                       PrintLayer_1(),
                                       nn.BatchNorm2d(64))

        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            
            # First transposed convolution
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            PrintLayer_1(),
            nn.BatchNorm2d(32),
            
            # Second transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            PrintLayer_1(),
            nn.BatchNorm2d(16),
            
            # Third transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            PrintLayer_1(),
            nn.BatchNorm2d(8),
            
            # fourth transposed convolution
            nn.ConvTranspose2d(8, 4, 3, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout),
            PrintLayer_1(),
            nn.BatchNorm2d(4),
            
            # fivth transposed convolution
            nn.ConvTranspose2d(4, 3, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            PrintLayer_1(),
            
        )
        
    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        
        x = self.unflatten(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        return x
    
#%%

def plot_ae_outputs(encoder,decoder,test_dataset,device,n=5):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[i][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()  