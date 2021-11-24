import torch
from UnetModel import UNet
from torch.optim import AdamW
from DataLoader import SegmentationDataset
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
def train(epochs,model, optimizer, scheduler,loss, dataloader):
    #model.cuda()
    for x in range(epochs):
        cum_loss = 0
        for element in dataloader:
            optimizer.zero_grad()
            t,y = element['img'].reshape(30,1,240,240).float() , element['GT'].reshape(30,1,64,64).float()
            out = model(t)
            loss_ = loss(out, y)
            loss_.backward()
            print(f"Loss is : {loss_}")
            cum_loss += loss_.item()
            optimizer.step()
            if scheduler:
                scheduler.step()


if __name__ == '__main__':

    model = UNet()
    optimizer = AdamW(model.parameters(), lr = 1e-5)
    loss = torch.nn.BCEWithLogitsLoss()
    path_segmentation = "02-Cancer_diagnosis/data/NoduleSegmentation/"
    path_diagnosi = "/home/user/PSIV/02-Cancer_diagnosis/data/Diagnosis"
    dataset = SegmentationDataset(path_segmentation, path_diagnosi)
    dataloader = DataLoader(dataset, batch_size = 30)
    train(10, model, optimizer, None, loss, dataloader)