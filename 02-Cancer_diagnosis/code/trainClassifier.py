import torch
from UnetModel import UNet
from torch.optim import AdamW
from DataLoader import DiagnosisEnd2End
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import torchvision
import wandb
from PIL import ImageOps
from Unet2 import UNET

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
from DiceLoss import DiceLoss
from Classifier import CancerClassifier
from results_analysis import results_analysis
BATCH_SIZE = 1
LR = 1e-3

seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def train(epochs,model, optimizer, scheduler,loss):
    model.cuda()
    wandb.init(project="Cancer Classification")
    config = wandb.config
    config.learning_rate = LR
    config.batch_size = BATCH_SIZE
    transform = T.Compose([T.Normalize(mean = (-640 ) , std=(380))])
    for x in range(epochs):
        cum_loss = 0
        diagnosis_path = "02-Cancer_diagnosis/data/Diagnosis/"
        nodules_path = "02-Cancer_diagnosis/data/Diagnosis/Radiolung_NoduleDiagnosis.csv"
        data_laoder = DiagnosisEnd2End(diagnosis_path, nodules_path)
        #print(len(data_laoder))
        dataloader = DataLoader(data_laoder, batch_size = BATCH_SIZE, drop_last=True)
        acc = 0
        predicts = []
        gt = []
        for element in tqdm(dataloader):
            optimizer.zero_grad()
            s = element['ROI'].shape[1] //2
            element['ROI'] = element['ROI'][:,s-7:s+7:,:]
            print(element['ROI'].shape)
            t,y = transform(element['ROI'].reshape(14,1,64,64).float()).cuda(), element['GT']
            print(t.shape)
            #t = t[:, s_-25:s_+25, :,:]
            out = model(t)
            #out_ = torch.argmax(out, axis=1).FLO()
            #y = y.reshape((BATCH_SIZE,64,64))
            acc += 1 if (out>=0.5) == int(y) else 0 
            
            
            loss_ = loss(out, y.float().cuda())
            loss_.backward()
            predicts.append((out>=0.5).item())
            gt.append(y.item())
            #print(f"Loss is : {loss_}")
            cum_loss += loss_.item()
            optimizer.step()
            wandb.log({'batch_loss': loss_.item()})
            if scheduler:
                scheduler.step()

        a,b =  results_analysis(gt, predicts)
        print(a)
        print(b)
        print(f"Acc on epoch {epochs} : {acc}")

        wandb.log({"epoch_loss": cum_loss})
        #out = otsu_transformation((255 * out).int().detach().cpu().numpy())
        #table = wandb.Table(data=[[element] for batch in out for channel in batch for row in channel for element in row], columns=["pixel_val"])
        #wandb.log({'my_histogram': wandb.plot.histogram(table, "pixel_val",
 	    #title="Prediction Score Distribution")})
        #plt.hist(
        #    out.flatten().detach().cpu().numpy()
        #)
        #plt.show()
        #out = torch.nn.Sigmoid()(out)
        #out[out < 0.5] = 0
        #out[out > 0.5] = 1
        #wandb.log({"model": wandb.Image(out),
        #            "GT": wandb.Image(y) })
        #print(f"Batched loss is {cum_loss}")


if __name__ == '__main__':

    model = CancerClassifier("02-Cancer_diagnosis/code/unet_segment.pth").cuda()
    optimizer = AdamW(model.parameters(), lr = LR)
    loss = torch.nn.BCELoss()
    #loss = DiceLoss()
    path_segmentation = "02-Cancer_diagnosis/data/NoduleSegmentation/"
    path_diagnosi = "/home/user/PSIV/02-Cancer_diagnosis/data/Diagnosis"
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=1)
    train(20, model, optimizer, None, loss)
    torch.save(model.state_dict(), 'unet_segment2.pth')