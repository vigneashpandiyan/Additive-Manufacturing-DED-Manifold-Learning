# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 11:58:30 2021

@author: srpv
"""

import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
import torchvision.transforms as transforms 
import torchvision 
from torchvision import datasets
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR

cuda = torch.cuda.is_available()
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize']=24
plt.rcParams['ytick.labelsize']=24
import matplotlib.font_manager
from sklearn import svm

from matplotlib import animation
from networks import *
from utils import *
from utils_SVM import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

print(__doc__)


#%% Folder pointing to the data

# Data--> https://polybox.ethz.ch/index.php/s/HUcJ7cJ18K0MrEn 

datadir = '../Data/'   #place in a folder
traindir = datadir + 'train/'
testdir = datadir + 'test/'
num_epochs = 1
batch_size = 500
n_classes = 6

#%% Data loader
data_transform = transforms.Compose([
        torchvision.transforms.Resize((480,320)),
        transforms.ToTensor()])
    
train_dataset = datasets.ImageFolder(root=traindir,transform=data_transform)
test_dataset = datasets.ImageFolder(root=testdir,transform=data_transform)


#%%

fig, axs = plt.subplots(5, 5, figsize=(10,10))
for ax in axs.flatten():
    # random.choice allows to randomly sample from a list-like object (basically anything that can be accessed with an index, like our dataset)
    img, label = random.choice(train_dataset)
    label=label+1
    img=img.permute(2,1,0)
    ax.imshow(np.array(img), cmap='gist_gray')
    # ax.imshow((img.cpu()))
    ax.set_title('P%d' % label,fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])
graph_title = "Conditions" + ".png" 
plt.savefig(graph_title, bbox_inches='tight',dpi=600)
plt.show()
# plt.tight_layout()

#%% Training 

classes = train_dataset.classes

# Set up data loaders

kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

#%%Set up the network and training parameters

torch.manual_seed(0)
encoder = Encoder(dropout=0.01)
decoder = Decoder(dropout=0.01)

'''Define the loss function'''
loss_fn = torch.nn.MSELoss()

'''Define an optimizer (both for the encoder and the decoder!)'''
lr= 0.001

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)


# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)


#%%


'''Training function'''
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    for image_batch, _ in train_loader: 
        image_batch = image_batch.to(device)
        
        encoded_data = encoder(image_batch)
        
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t train loss : %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

history={'train_loss':[]}

for epoch in range(num_epochs):

    train_loss = train_epoch(encoder,decoder,device,train_loader,loss_fn,optim)
    history['train_loss'].append(train_loss)


train_plot = 'train_losses'+'_'+ '.npy'
np.save(train_plot,history['train_loss'], allow_pickle=True)


plt.figure(figsize=(10,7))
plt.plot(np.load('train_losses_.npy'),'b', label='Autoencoder training',linewidth =4.0)
# plt.semilogy(history['val_loss'], label='Valid')
plt.xlabel('Epoch',fontsize=25)
plt.ylabel('Average loss',fontsize=25)
#plt.grid()
plt.legend( loc='upper right',fontsize=30)
plt.savefig('Loss_Conv.png', dpi=600,bbox_inches='tight')
plt.show()
plt.clf()


#%%
# Plot Embeddings

def rotate(angle):
      ax.view_init(azim=angle)
   

train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, encoder)
train_embeddings_baseline=train_embeddings_baseline.astype(np.float64)
train_labels_baseline=train_labels_baseline.astype(np.float64)
train_embeddings = 'train_embeddings'+ '.npy'
train_labelsname = 'train_labels'+'.npy'
np.save(train_embeddings,train_embeddings_baseline, allow_pickle=True)
np.save(train_labelsname,train_labels_baseline, allow_pickle=True)
graph_name= 'Tsne Testing_2D'+'.png'
tsne,tsne_fit_train,target_train=TSNEplot(train_embeddings_baseline,train_labels_baseline,graph_name,20)
plot_embeddings(tsne_fit_train, target_train,'train',train_dataset)


graph_name='Training_Feature_3D'+'.png'
ax,fig=Three_embeddings(tsne_fit_train, target_train,graph_name,ang=320)
gif1_name= str('Training_Feature')+'.gif'


      
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(gif1_name, writer=animation.PillowWriter(fps=20))



#%%
val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, encoder)
val_embeddings_baseline=val_embeddings_baseline.astype(np.float64)
val_labels_baseline=val_labels_baseline.astype(np.float64)
val_embeddings = 'val_embeddings'+'.npy'
val_labelsname = 'val_labels'+'.npy'
np.save(val_embeddings,val_embeddings_baseline, allow_pickle=True)
np.save(val_labelsname,val_labels_baseline, allow_pickle=True)
tsne,tsne_fit_val,target_val=TSNEplot(val_embeddings_baseline,val_labels_baseline,graph_name,20)
plot_embeddings(tsne_fit_val, target_val,'test',test_dataset)
#%%

graph_name='Testing_Feature_3D'+'.png'
ax,fig=Three_embeddings(tsne_fit_val, target_val,graph_name,ang=320)
gif1_name= str('Testing_Feature')+'.gif'

      
angle = 3
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
ani.save(gif1_name, writer=animation.PillowWriter(fps=20))

#%%SVM implementation
Material = "train"
classname="P4"


data,labels=Normal_Regime(Material, classname)
Outliers,_=Abnormal_Regime(Material, classname)

data=data.to_numpy()
Outliers=Outliers.to_numpy()

from sklearn.model_selection import train_test_split# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=66)

#%%


# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(Outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size


print(
    "error train: %d/%d ; errors novel regular: %d/%d ; "
    "errors novel abnormal: %d/%d"
    % (n_error_train,len(X_train), n_error_test,len(X_test), n_error_outliers,len(Outliers)))

#%%


fig = plt.figure(figsize=(7,7), dpi=800)
plt.rcParams["legend.markerscale"] = 2
s = 60
c = plt.scatter(Outliers[:, 0], Outliers[:, 3], c="gold", s=s, edgecolors="k",alpha=0.2)
b1 = plt.scatter(X_train[:, 0], X_train[:, 3], c="blue", s=s, edgecolors="k")
b2 = plt.scatter(X_test[:, 0], X_test[:, 3], c="lightblue", s=s, edgecolors="k")

plt.axis("tight")
plt.xlabel ('Dimension 1', labelpad=5,fontsize=20)
plt.ylabel ('Dimension 2', labelpad=5,fontsize=20)
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.legend(
    [ b1, b2, c],
    [
        
        "Training observations (P4)",
        "P4 observations for testing",
        "Outlier observations (P1,P2, P3, P5, P6)",
    ],
    loc="lower left",
    prop=matplotlib.font_manager.FontProperties(size=15),
)

plt.savefig('graph_name_1.png', bbox_inches='tight',dpi=800)
plt.show()
