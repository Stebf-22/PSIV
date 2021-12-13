import torch
import numpy as np
import torchvision
import random

class BatchAugemntation:


    def __init__(self, n_input, n_output, available_transformations, probs=None):

        if probs:
            self.probs = [1/(len(available_transformations)) for x in available_transformations]

        self.n_input = n_input
        self.n_output = n_output
        self.transoforms  = available_transformations
    
    def augment(self, data):
        img, GT = data
        for transformation, prob in zip(self.transoforms, self.probs):
            if random.random() <= prob:
                img = transformation(img)
                GT = transformation(GT)
        
        return img, GT
    
    def augment_restricted(self,batch ):
        '''
        Special Augementation, characteristics:
            - 1 image -> 1 augmentated images ( mas 2x batch)
            - Same image cannot be augemented two times
        '''

        assert len(batch) == self.n_input, "Make sure you are initializing the class with the correct n_input parameters( batch_size)"
        num_imgs = self.n_output - self.n_input
        assert len(batch) >= num_imgs, "The idea is to just augment one image per batch"
        already_augmented = set()
        img, GT = batch['ROI'], batch['GT']
        i = 0
        values_list_fornmat = [ x for x in batch['ROI']]
        gt_list_format = [x for x in batch['GT']]
        while i < num_imgs:
            idx = random.randint(0, len(batch))
            if idx in already_augmented:
                continue
            already_augmented.add(idx)
            img_out, GT_out = self.augment((img, GT))
            values_list_fornmat.append(img_out)
            gt_list_format.append(GT_out)
            i+=1
        return {'ROI': torch.stack(values_list_fornmat) , 'GT': torch.stack(gt_list_format)}



if __name__ == '__main__':

    transforms = torchvision.Compose(
        
    )



        
            
            