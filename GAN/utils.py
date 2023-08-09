import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
cuda = torch.cuda.is_available()
from sklearn.manifold import TSNE
import pandas as pd

colors = [ 'purple','orange','green','red', 'blue', 'cyan']
mnist_classes = ['P1', 'P2', 'P3', 'P4','P5','P6']
marker= ["d","s","*",">","X","o"]
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams["legend.markerscale"] = 2.1


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 512))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def TSNEplot(z_run,test_labels,graph_name,perplexity):
    
    output = z_run
    #array of latent space, features fed rowise
    
    target = test_labels
    #groundtruth variable
    
    print('target shape: ', target.shape)
    print('output shape: ', output.shape)
    print('perplexity: ',perplexity)
    

    group=target
    group = np.ravel(group)
        
    RS=np.random.seed(1974)
    tsne = TSNE(n_components=3, random_state=RS, perplexity=perplexity)
    tsne_fit = tsne.fit_transform(output)
        
    return tsne,tsne_fit,target


def plot_embeddings(embeddings, targets,name,train_dataset, xlim=None, ylim=None):
    plt.figure(figsize=(9,6))
    for i in range(len(train_dataset.classes)):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=1, color=colors[i],marker=marker[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)
    graph_title = "Feature space distribution " + str(name)
    plt.xlabel ('Dimension 1', labelpad=10,fontsize = 15)
    plt.ylabel ('Dimension 2', labelpad=10,fontsize = 15)
    plt.savefig(graph_title, bbox_inches='tight',dpi=600)
    plt.show()
    
def Three_embeddings(embeddings, targets,graph_name,ang, xlim=None, ylim=None):
    group=targets
    
    df2 = pd.DataFrame(group) 
    df2.columns = ['Categorical']
    df2=df2['Categorical'].replace(0,'P1')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(1,'P2')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(2,'P3')
    df2 = pd.DataFrame(df2)
    df2=df2['Categorical'].replace(3,'P4')
    df2 = pd.DataFrame(df2) 
    df2=df2['Categorical'].replace(4,'P5')
    df2 = pd.DataFrame(df2) 
    df2=df2['Categorical'].replace(5,'P6')
    group = pd.DataFrame(df2) 
    
    group=group.to_numpy()
    group = np.ravel(group)
    
    
    x1=embeddings[:, 0]
    x2=embeddings[:, 1]
    x3=embeddings[:, 2]
    
    
    df = pd.DataFrame(dict(x=x1, y=x2,z=x3, label=group))
    groups = df.groupby('label')
    uniq = list(set(df['label']))
    uniq=np.sort(uniq)
    #uniq=["0","1","2","3"]
    
    
    fig = plt.figure(figsize=(12,6), dpi=100)
    fig.set_facecolor('white')
   
    ax = plt.axes(projection='3d')
    
    ax.grid(False)
    ax.view_init(azim=ang)#115
    marker= ["d","s","*",">","X","o"]
    color = [ 'purple','orange','green','red', 'blue', 'cyan']
    
    ax.set_facecolor('white') 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    graph_title = "Feature space distribution"
    
    j=0
    for i in uniq:
        print(i)
        indx = group == i
        a=x1[indx]
        b=x2[indx]
        c=x3[indx]
        ax.plot(a, b, c ,color=color[j],label=uniq[j],marker=marker[j],linestyle='',ms=7)
        j=j+1
     
    plt.xlabel ('Dimension 1', labelpad=20,fontsize = 15)
    plt.ylabel ('Dimension 2', labelpad=20,fontsize = 15)
    ax.set_zlabel('Dimension 3',labelpad=20,fontsize = 15)
    plt.title(str(graph_title),fontsize = 15)
    
    plt.legend(markerscale=20)
    plt.locator_params(nbins=6)
    plt.legend(loc='upper left',frameon=False)
    plt.savefig(graph_name, bbox_inches='tight',dpi=400)
    plt.show()
    return ax,fig