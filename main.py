'''
@ author: Jiali Duan
@ function: Saak Transform
@ Date: 10/29/2017
@ To do: parallelization
'''

# load libs
import torch
import argparse
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from data.datasets import MNIST
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product

# argument parsing
print torch.__version__
batch_size=1
test_batch_size=1
kwargs={}
train_loader=data_utils.DataLoader(MNIST(root='./data',train=True,process=False,transform=transforms.Compose([
    transforms.Scale((32,32)),
    transforms.ToTensor(),
])),batch_size=batch_size,shuffle=True,**kwargs)


test_loader=data_utils.DataLoader(MNIST(root='./data',train=False,process=False,transform=transforms.Compose([
    transforms.Scale((32,32)),
    transforms.ToTensor(),
])),batch_size=test_batch_size,shuffle=True,**kwargs)



# show sample
def show_sample(inv):
    inv_img=inv.data.numpy()[0][0]
    plt.imshow(inv_img)
    plt.gray()
    plt.savefig('./image/demo.png')
   # plt.show()

'''
@ For demo use, only extracts the first 1000 samples
'''
def create_numpy_dataset():
    datasets = []
    for data in train_loader:
        data_numpy = data[0].numpy()
        data_numpy = np.squeeze(data_numpy)
        datasets.append(data_numpy)

    datasets = np.array(datasets)
    datasets=np.expand_dims(datasets,axis=1)
    print 'Numpy dataset shape is {}'.format(datasets.shape)
    return datasets[:1000]



'''
@ data: flatten patch data: (14*14*60000,1,2,2)
@ return: augmented anchors
'''
def PCA_and_augment(data_in):
    # data reshape
    data=np.reshape(data_in,(data_in.shape[0],-1))
    print 'PCA_and_augment: {}'.format(data.shape)
    # mean removal
    mean = np.mean(data, axis=0)
    datas_mean_remov = data - mean
    print 'PCA_and_augment meanremove shape: {}'.format(datas_mean_remov.shape)

    # PCA, retain all components
    pca=PCA()
    pca.fit(datas_mean_remov)
    comps=pca.components_

    # augment, DC component doesn't
    comps_aug=[vec*(-1) for vec in comps[:-1]]
    comps_complete=np.vstack((comps,comps_aug))
    print 'PCA_and_augment comps_complete shape: {}'.format(comps_complete.shape)
    return comps_complete



'''
@ datasets: numpy data as input
@ depth: determine shape, initial: 0
'''

def fit_pca_shape(datasets,depth):
    factor=np.power(2,depth)
    length=32/factor
    print 'fit_pca_shape: length: {}'.format(length)
    idx1=range(0,length,2)
    idx2=[i+2 for i in idx1]
    print 'fit_pca_shape: idx1: {}'.format(idx1)
    data_lattice=[datasets[:,:,i:j,k:l] for ((i,j),(k,l)) in product(zip(idx1,idx2),zip(idx1,idx2))]
    data_lattice=np.array(data_lattice)
    print 'fit_pca_shape: data_lattice.shape: {}'.format(data_lattice.shape)

    #shape reshape
    data=np.reshape(data_lattice,(data_lattice.shape[0]*data_lattice.shape[1],data_lattice.shape[2],2,2))
    print 'fit_pca_shape: reshape: {}'.format(data.shape)
    return data


'''
@ Prepare shape changes. 
@ return filters and datasets for convolution
@ aug_anchors: [7,4] -> [7,input_shape,2,2]
@ output_datas: [60000*num_patch*num_patch,channel,2,2]

'''
def ret_filt_patches(aug_anchors,input_channels):
    shape=aug_anchors.shape[1]/4
    num=aug_anchors.shape[0]
    filt=np.reshape(aug_anchors,(num,shape,4))

    # reshape to kernels, (7,shape,2,2)
    filters=np.reshape(filt,(num,shape,2,2))

    # reshape datasets, (60000*shape*shape,shape,28,28)
    # datasets=np.expand_dims(dataset,axis=1)

    return filters



'''
@ input: numpy kernel and data
@ output: conv+relu result
'''
def conv_and_relu(filters,datasets,stride=2):
    # torch data change
    filters_t=torch.from_numpy(filters)
    datasets_t=torch.from_numpy(datasets)

    # Variables
    filt=Variable(filters_t).type(torch.FloatTensor)
    data=Variable(datasets_t).type(torch.FloatTensor)

    # Convolution
    output=F.conv2d(data,filt,stride=stride)

    # Relu
    relu_output=F.relu(output)

    return relu_output,filt



'''
@ One-stage Saak transform
@ input: datasets [60000,channel, size,size]
'''
def one_stage_saak_trans(datasets=None,depth=0):


    # load dataset, (60000,1,32,32)
    # input_channel: 1->7
    print 'one_stage_saak_trans: datasets.shape {}'.format(datasets.shape)
    input_channels=datasets.shape[1]

    # change data shape, (14*60000,4)
    data_flatten=fit_pca_shape(datasets,depth)

    # augmented components
    comps_complete=PCA_and_augment(data_flatten)
    print 'one_stage_saak_trans: comps_complete: {}'.format(comps_complete.shape)

    # get filter and datas, (7,1,2,2) (60000,1,32,32)
    filters=ret_filt_patches(comps_complete,input_channels)
    print 'one_stage_saak_trans: filters: {}'.format(filters.shape)

    # output (60000,7,14,14)
    relu_output,filt=conv_and_relu(filters,datasets,stride=2)

    data=relu_output.data.numpy()
    print 'one_stage_saak_trans: output: {}'.format(data.shape)
    return data,filt,relu_output



'''
@ Multi-stage Saak transform
'''
def multi_stage_saak_trans():
    filters = []
    outputs = []

    data=create_numpy_dataset()
    dataset=data
    num=0
    img_len=data.shape[-1]
    while(img_len>=2):
        num+=1
        img_len/=2


    for i in range(num):
        print '{} stage of saak transform: '.format(i)
        data,filt,output=one_stage_saak_trans(data,depth=i)
        filters.append(filt)
        outputs.append(output)
        print ''


    return dataset,filters,outputs

'''
@ Reconstruction from the second last stage
@ In fact, reconstruction can be done from any stage
'''
def toy_recon(outputs,filters):
    outputs=outputs[::-1][1:]
    filters=filters[::-1][1:]
    num=len(outputs)
    data=outputs[0]
    for i in range(num):
        data = F.conv_transpose2d(data, filters[i], stride=2)

    return data


if __name__=='__main__':
    dataset,filters,outputs=multi_stage_saak_trans()
    data=toy_recon(outputs,filters)
    show_sample(data)






