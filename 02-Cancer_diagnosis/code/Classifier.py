from Unet2 import UNET
import torch
from torch import nn
from ImageTransformations import otsu

class CancerClassifier(nn.Module):
    def __init__(self, unet_path, segemnt =True ) -> None:
        super().__init__()
        dict_ = torch.load(unet_path)
        self.unet = UNET(1,1).cuda().eval()
        self.unet.load_state_dict(dict_)
        self.unet = self.unet.cpu()
        self.relu = nn.ReLU()
        ### Block 1
        self.conv1 = nn.Conv2d(14, 128 , 3 )
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256 , 3 )
        self.batch_norm2 = nn.BatchNorm2d(256)

        #Block 2: Amplifying POV
        self.conv3 = nn.Conv2d(256, 512 , 5 )
        self.batch_norm3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 4 , 5 )
        self.batch_norm4 = nn.BatchNorm2d(4)
        self.classifier = nn.Linear(10816, 1)

        self.segment = segemnt
    def forward(self, x):
        if self.segment:
            segmented = torch.from_numpy(otsu(self.unet(x).reshape(1,14,64,64).detach().cpu().numpy())).float().cuda()
        else: segmented =x
        feat = self.relu(self.batch_norm1(self.conv1(segmented)))
        feat2 = self.relu(self.batch_norm2(self.conv2(feat)))
        feat3 = self.relu(self.batch_norm3(self.conv3(feat2)))
        return nn.Sigmoid()(self.classifier(self.batch_norm4(self.conv4(feat3)).flatten()))




if __name__ == '__main__':

    df = CancerClassifier('unet_segment.pth')
    test = torch.zeros(50,1,64,64)
    print(df(test).shape)


