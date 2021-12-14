import torch
from CovAttention import ConvAtt
from DataLoader import DiagnosisEnd2End
from torch.utils.data import DataLoader
def train(model , optimizer, loss, scheduler, dataloaderT, epochs):
    for e in range(epochs):
        cum = 0
        for batch in dataloaderT:
            optimizer.zero_grad()
            imgs, gt = batch['ROI'], batch['GT'].cuda()
            imgs  = imgs.reshape(-1,1,64,64).float().cuda()
            deep = imgs.shape[0]
            imgs = imgs[deep//2 - 7 : deep//2 +7,:,:,:]
            out = model(imgs)

           # print(out.argmax().shape, gt.shape)
            loss_ = torch.nn.functional.binary_cross_entropy(out.view(1), gt.float())
            loss_.backward()
            #cum += loss_.cpu().item()
            optimizer.step()
            if scheduler:
                scheduler.step()


if __name__ == '__main__':

    conv_att = ConvAtt(1, 50, 50, 5).cuda()
    optimizer = torch.optim.AdamW(conv_att.parameters(), lr = 1e-3)
    looss = None
    path_segmentation = "02-Cancer_diagnosis/data/Diagnosis/"
    nodules_path = "02-Cancer_diagnosis/data/Diagnosis/Radiolung_NoduleDiagnosis.csv"
    dataset = DataLoader(DiagnosisEnd2End(path_segmentation, nodules_path), batch_size=1)
    train(conv_att, optimizer, looss, None, dataset, 1 )



