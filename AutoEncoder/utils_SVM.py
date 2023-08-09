from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import pandas as pd
from plotly.graph_objs import *
import plotly
import torch
# from prettytable import PrettyTable


def Normal_Regime(Material, classname):
    
    classfile_1 = str(Material)+'_labels'+'.npy'
    rawfile_1 = str(Material)+'_embeddings'+'.npy'
    target_1= np.load(classfile_1)
    Features_1 = np.load(rawfile_1)
    print(Features_1.shape)
    
    df1 = pd.DataFrame(Features_1)  
    df1=df1[df1.select_dtypes(include=['number']).columns] * 1
    
    df2 = pd.DataFrame(target_1) 
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
   
    
    df_1=pd.concat([df1,df2], axis=1)
    new_columns = list(df_1.columns)
    new_columns[-1] = 'target'
    df_1.columns = new_columns
    df_1.target.value_counts()
    df_1 = df_1.sample(frac=1.0)
    class_name = classname 
    Normal=class_name
    df_1 = df_1[df_1.target == str(Normal)]
    print(df_1.shape)
    
    labels=df_1.iloc[:,-1]
    #validation_labels=validation_labels.to_numpy().astype(np.float64)
    df_1 = df_1.drop(labels='target', axis=1)
    print(df_1.shape)
    print(labels.shape)
    
    labels = pd.DataFrame(labels) 
    labels.columns = ['Categorical']
    labels=labels['Categorical'].replace('P2',-1)
    
    return df_1,labels


def Abnormal_Regime(Material, classname):
    
    classfile_1 = str(Material)+'_labels'+'.npy'
    rawfile_1 = str(Material)+'_embeddings'+'.npy'
    target_1= np.load(classfile_1)
    Features_1 = np.load(rawfile_1)
    print(Features_1.shape)
    
    df1 = pd.DataFrame(Features_1)  
    df1=df1[df1.select_dtypes(include=['number']).columns] * 1
    
    df2 = pd.DataFrame(target_1) 
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
   
    
    df_1=pd.concat([df1,df2], axis=1)
    new_columns = list(df_1.columns)
    new_columns[-1] = 'target'
    df_1.columns = new_columns
    df_1.target.value_counts()
    df_1 = df_1.sample(frac=1.0)
    class_name = classname 
    Normal=class_name
    df_1 = df_1[df_1.target != str(Normal)]
    print(df_1.shape)
    
    labels=df_1.iloc[:,-1]
    #validation_labels=validation_labels.to_numpy().astype(np.float64)
    df_1 = df_1.drop(labels='target', axis=1)
    print(df_1.shape)
    print(labels.shape)
    
    labels = pd.DataFrame(labels) 
    labels.columns = ['Categorical']
    labels=labels['Categorical'].replace('P1',1)
    labels = pd.DataFrame(labels)
    labels=labels['Categorical'].replace('P3',1)
    labels = pd.DataFrame(labels)
    labels=labels['Categorical'].replace('P4',1)
    labels = pd.DataFrame(labels)
    labels=labels['Categorical'].replace('P5',1)
    labels = pd.DataFrame(labels)
    labels=labels['Categorical'].replace('P6',1)
    labels = pd.DataFrame(labels)
    
    return df_1,labels


def Normal_tsne_Regime(Material, classname):
    
    classfile_1 = str(Material)+'_labels'+'.npy'
    rawfile_1 = str(Material)+'_embeddings'+'.npy'
    target_1= np.load(classfile_1)
    Features_1 = np.load(rawfile_1)
    print(Features_1.shape)
    
    df1 = pd.DataFrame(Features_1)  
    df1=df1[df1.select_dtypes(include=['number']).columns] * 1
    
    df2 = pd.DataFrame(target_1) 
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
   
    
    df_1=pd.concat([df1,df2], axis=1)
    new_columns = list(df_1.columns)
    new_columns[-1] = 'target'
    df_1.columns = new_columns
    df_1.target.value_counts()
    df_1 = df_1.sample(frac=1.0)
    class_name = classname 
    Normal=class_name
    df_1 = df_1[df_1.target == str(Normal)]
    print(df_1.shape)
    
    labels=df_1.iloc[:,-1]
    #validation_labels=validation_labels.to_numpy().astype(np.float64)
    df_1 = df_1.drop(labels='target', axis=1)
    print(df_1.shape)
    print(labels.shape)
    
    labels = pd.DataFrame(labels) 
    labels.columns = ['Categorical']
    labels=labels['Categorical'].replace('P2',-1)
    
    return df_1,labels
    


def Abnormal_tsne_Regime(Material, classname):
    
    classfile_1 = str(Material)+'_labels'+'.npy'
    rawfile_1 = str(Material)+'_embeddings'+'.npy'
    target_1= np.load(classfile_1)
    Features_1 = np.load(rawfile_1)
    print(Features_1.shape)
    
    df1 = pd.DataFrame(Features_1)  
    df1=df1[df1.select_dtypes(include=['number']).columns] * 1
    
    df2 = pd.DataFrame(target_1) 
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
   
    
    df_1=pd.concat([df1,df2], axis=1)
    new_columns = list(df_1.columns)
    new_columns[-1] = 'target'
    df_1.columns = new_columns
    df_1.target.value_counts()
    df_1 = df_1.sample(frac=1.0)
    class_name = classname 
    Normal=class_name
    df_1 = df_1[df_1.target != str(Normal)]
    print(df_1.shape)
    
    labels=df_1.iloc[:,-1]
    #validation_labels=validation_labels.to_numpy().astype(np.float64)
    df_1 = df_1.drop(labels='target', axis=1)
    print(df_1.shape)
    print(labels.shape)
    
    labels = pd.DataFrame(labels) 
    labels.columns = ['Categorical']
    labels=labels['Categorical'].replace('P1',1)
    labels = pd.DataFrame(labels)
    labels=labels['Categorical'].replace('P3',1)
    labels = pd.DataFrame(labels)
    labels=labels['Categorical'].replace('P4',1)
    labels = pd.DataFrame(labels)
    labels=labels['Categorical'].replace('P5',1)
    labels = pd.DataFrame(labels)
    labels=labels['Categorical'].replace('P6',1)
    labels = pd.DataFrame(labels)
    
   
    return df_1,labels
