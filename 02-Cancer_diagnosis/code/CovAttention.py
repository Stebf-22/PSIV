from torch import nn
import torch

class ConvBlock(nn.Module):

    def __init__(self, in_ch=1, out_dim=20):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch , 64,5)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.conv2 =  nn.Conv2d(64 , 128,3)
        self.conv3  = nn.Conv2d(128, 64,3)
        self.conv4 = nn.Conv2d(64,1,3)
        self.out = nn.Linear(2916,out_dim)
    
    def forward(self, img):
        out = self.relu(self.batch_norm1(self.conv1(img)))
        out = self.relu(self.batch_norm2(self.conv2(out)))
        out = self.relu(self.batch_norm3(self.conv3(out)))
        out = self.relu(self.batch_norm4(self.conv4(out)))
        #print(img.shape, out.shape)
        return self.out(out.reshape(img.shape[0], -1))
    

class Attention(nn.Module):

    def __init__(self,embed_dim , n_heads = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.att = torch.nn.MultiheadAttention(self.embed_dim, self.n_heads)
        self.last = nn.Linear(14 * self.embed_dim, 1)

    def forward(self, features):
      
        attn_output, weights = self.att(features, features, features)
        charachteristics = attn_output.reshape(1, -1)
        return self.last(charachteristics)

class ConvAtt(nn.Module):

    def __init__(self, in_ch, out_dim, embed_dim, n_heads):
        super().__init__()
        self.convol = ConvBlock(in_ch, out_dim)
        self.att = Attention(embed_dim, n_heads)

    def forward(self, img):
        conv_out = self.convol(img)
        n_depth = conv_out.shape[0]
        conv_out = conv_out.reshape(n_depth, 1, -1)
        return self.att(conv_out)


if __name__ == '__main__':
    '''model = ConvBlock(out_dim=50)
    e = model(torch.zeros(14,1,64,64))
    e = e.reshape(14,1,50)
    model_att = Attention( 50, 5)
    print(model_att(e))'''
    t = ConvAtt(1,50, 50 , 5)
    print(t(torch.zeros(14,1,64,64)))

