from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
import numpy as np
import torch
from torch import nn
'''image = np.load("/home/user/PSIV/02-Cancer_diagnosis/data/SegmentedDiagnosi/Paciente_1_TC_9/segmented.npy",allow_pickle=True)[:,:,:3]
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch32-384')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384')
model.__dict__['_modules']['classifier'] = nn.Linear(1024, 2)
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
print(outputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])'''



class ViTHandler(nn.Module):

    def __init__(self, model_name = 'google/vit-large-patch32-384', out_size = 2, transfer=False):
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.__dict__['_modules']['classifier'] = nn.Linear(1024, 2)
        self.__finetune_freezing()
        
    def __finetune_freezing(self):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.classifier.requires_grad = True
    
    def forward(self, img):
        print(img.shape)
        t = torch.from_numpy(img)
        t = t.permute(2,1,0)
        print(t.shape)
        to_eval = torch.zeros(t.shape[0]//3, 3, t.shape[1], t.shape[2])
        idx = 0
        for x in range(0,t.shape[0]//3):
            print(x,idx)
            to_eval[x] = t[idx:idx+3,:,:]
            idx+=3
        return self.model(to_eval)
if __name__ == '__main__':
    model = ViTHandler()

    print(model(np.zeros((384,384,201))).logits.flatten())