from numpy import result_type
import torch
import cv2
import os
import random
from .IamgesTransformations import doTransformations
import matplotlib.pyplot as plt
import torchvision
from .IamgesTransformations import resize
import numpy as np

class DatasetAugmentated(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, image_path, max_sample):
        'Initialization'
        self.labels = image_path
        images = os.listdir(image_path)
        full_path = [image_path + x for x in images]
        character = [x.split('_')[1][0] for x in images]
        self.images = {char: [] for  char in set(character)}

        for char,x in zip(character,full_path):
            self.images[char].append(cv2.imread(x, 0 ))
        self.max_number_samples = max_sample
        self.generate_order()
        self.char_to_indx = {x: y for x,y in zip(self.images,range(len(self.images)) )}
  def __len__(self):    
        'Denotes the total number of samples'
        return self.max_number_samples

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if index%35 == 0:
            self.generate_order()
        
        char = self.order[index % len(self.images)]
        indx = random.randint(0,len(self.images[char]) -1 )
        #print(self.images[char].shape)
        # Load data and get label
        #print(self.images[char])
  
        img = resize((doTransformations(self.images[char][indx])))[np.newaxis, : ].astype(np.float32)
        return img, self.char_to_indx[char]

  def generate_order(self):
       x = list(self.images.keys())
       random.shuffle(x)
       self.order = x
       return 

if __name__ == '__main__':

    dataset = DatasetAugmentated('/content/drive/MyDrive/MatriculasDatset/images3/', 10000)
    dic = dataset.char_to_indx
    print(dataset.char_to_indx)
    a,b = '',''
    for x in dic:
        a+=str(x)
        b+=str(dic[x])
    print(a)
    print(b)
    '''
    plt.imshow(dataset.images['A'])
    print(len(dataset.images.keys()))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 5)
    for batch in dataloader:
         img, t = batch
         #for x in range(10):
          #     plt.imshow(img[x].reshape(60,30))
         #plt.show()
         #grid_img = torchvision.utils.make_grid(img[:10], nrow=5)
         #plt.imshow(grid_img.permute(1, 2, 0))
         #plt.show()
         break'''

