import torch
from UnetModel import UNet
from torch.optim import AdamW
from DataLoader import SegmentationDataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import torchvision
import wandb
from PIL import ImageOps
from Unet2 import UNET
from ImageTransformations import otsu_transformation
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
BATCH_SIZE = 32
LR = 1e-3

seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def train(epochs,model, optimizer, scheduler,loss):
    #model.cuda()
    wandb.init(project="Cancer Classification")
    config = wandb.config
    config.learning_rate = LR
    config.batch_size = BATCH_SIZE
    transform = T.Compose([T.Normalize(mean = (-640 ) , std=(380))])
    for x in range(epochs):
        cum_loss = 0
        dataset = SegmentationDataset(path_segmentation, path_diagnosi)
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, drop_last=True)
        for element in tqdm(dataloader):
            optimizer.zero_grad()
            t,y = transform(element['img'].reshape(BATCH_SIZE,1,64,64).float()).cuda() , element['GT'].reshape(BATCH_SIZE,1,64,64).float().cuda()
            out = model(t)
            loss_ = loss(out, y)
            loss_.backward()
            #print(f"Loss is : {loss_}")
            cum_loss += loss_.item()
            optimizer.step()
            wandb.log({'batch_loss': loss_.item()})
            if scheduler:
                scheduler.step()
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
        wandb.log({"model": wandb.Image(out),
                    "GT": wandb.Image(y) })
        #print(f"Batched loss is {cum_loss}")


if __name__ == '__main__':

    model = UNET(1,1).cuda()
    optimizer = AdamW(model.parameters(), lr = LR)
    loss = torch.nn.BCEWithLogitsLoss()
    path_segmentation = "02-Cancer_diagnosis/data/NoduleSegmentation/"
    path_diagnosi = "/home/user/PSIV/02-Cancer_diagnosis/data/Diagnosis"
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=1)
    train(20, model, optimizer, None, loss)