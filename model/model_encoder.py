import torch
from torch import nn
import torchvision.models as models
from einops import rearrange
from model.model_decoder import resblock
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, network):
        super(Encoder, self).__init__()
        self.network = network
        if self.network=='alexnet': #256,7,7
            cnn = models.alexnet(pretrained=True)
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg11': #512,1/32H,1/32W
            cnn = models.vgg11(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg16': #512,1/32H,1/32W
            cnn = models.vgg16(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='vgg19':#512,1/32H,1/32W
            cnn = models.vgg19(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='inception': #2048,6,6
            cnn = models.inception_v3(pretrained=True, aux_logits=False)  
            modules = list(cnn.children())[:-3]
        elif self.network=='resnet18': #512,1/32H,1/32W
            cnn = models.resnet18(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet34': #512,1/32H,1/32W
            cnn = models.resnet34(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet50': #2048,1/32H,1/32W
            cnn = models.resnet50(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet101':  #2048,1/32H,1/32W
            cnn = models.resnet101(pretrained=True)  
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(cnn.children())[:-2]
        elif self.network=='resnet152': #512,1/32H,1/32W
            cnn = models.resnet152(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext50_32x4d': #2048,1/32H,1/32W
            cnn = models.resnext50_32x4d(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='resnext101_32x8d':#2048,1/256H,1/256W
            cnn = models.resnext101_32x8d(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='densenet121': #no AdaptiveAvgPool2d #1024,1/32H,1/32W
            cnn = models.densenet121(pretrained=True) 
            modules = list(cnn.children())[:-1] 
        elif self.network=='densenet169': #1664,1/32H,1/32W
            cnn = models.densenet169(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='densenet201': #1920,1/32H,1/32W
            cnn = models.densenet201(pretrained=True)  
            modules = list(cnn.children())[:-1]
        elif self.network=='regnet_x_400mf': #400,1/32H,1/32W
            cnn = models.regnet_x_400mf(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='regnet_x_8gf': #1920,1/32H,1/32W
            cnn = models.regnet_x_8gf(pretrained=True)  
            modules = list(cnn.children())[:-2]
        elif self.network=='regnet_x_16gf': #2048,1/32H,1/32W
            cnn = models.regnet_x_16gf(pretrained=True) 
            modules = list(cnn.children())[:-2]

        self.cnn = nn.Sequential(*modules)
        # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, imageA, imageB):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        feat1 = self.cnn(imageA)  # (batch_size, 2048, image_size/32, image_size/32)
        feat2 = self.cnn(imageB)

        return feat1, feat2

    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.cnn.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.cnn.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadAtt(nn.Module):
    def __init__(self, dim_q, dim_kv, attention_dim, heads = 8, dropout = 0.):
        super(MultiHeadAtt, self).__init__()
        project_out = not (heads == 1 and attention_dim == dim_kv)
        self.heads = heads
        self.scale = (attention_dim // self.heads) ** -0.5

        self.to_q = nn.Linear(dim_q, attention_dim, bias = False)
        self.to_k = nn.Linear(dim_kv, attention_dim, bias = False)
        self.to_v = nn.Linear(dim_kv, attention_dim, bias = False)       
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(attention_dim, dim_q),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2, x3):
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_k(x3)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h = self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)#(b,n,dim)

class Transformer(nn.Module):
    def __init__(self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout = 0., norm_first = False):
        super(Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads = heads, dropout = dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim, dropout = dropout)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

    def forward(self, x1, x2, x3):
        if self.norm_first:
            x = self.att(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + x1
            x = self.feedforward(self.norm2(x)) + x
        else:
            x = self.norm1(self.att(x1, x2, x3) + x1)
            x = self.norm2(self.feedforward(x) + x)

        return x

class AttentiveEncoder(nn.Module):
    """
    One visual transformer block
    """
    def __init__(self, n_layers, feature_size, heads, hidden_dim, attention_dim = 512, dropout = 0.):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        
        self.h_embedding = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(w_feat, int(channels/2))
        self.selftrans = nn.ModuleList([])
        for i in range(n_layers):                 
            self.selftrans.append(nn.ModuleList([
                Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False),
                Transformer(channels*2, channels*2, heads, attention_dim, hidden_dim, dropout, norm_first=False),
            ]))

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img1, img2):
        batch, c, h, w = img1.shape
        pos_h = torch.arange(h).cuda()
        pos_w = torch.arange(w).cuda()
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                       embed_h.unsqueeze(1).repeat(1, w, 1)], 
                                       dim = -1)                            
        pos_embedding = pos_embedding.permute(2,0,1).unsqueeze(0).repeat(batch, 1, 1, 1)
        img1 = img1 + pos_embedding
        img2 = img2 + pos_embedding
        img1 = img1.view(batch, c, -1).transpose(-1, 1)#batch, hw, c
        img2 = img2.view(batch, c, -1).transpose(-1, 1)
        img_sa1, img_sa2 = img1, img2

        for (l, m) in self.selftrans:           
            img_sa1 = l(img_sa1, img_sa1, img_sa1) + img_sa1
            img_sa2 = l(img_sa2, img_sa2, img_sa2) + img_sa2
            img = torch.cat([img_sa1, img_sa2], dim = -1)
            img = m(img, img, img)
            img_sa1 = img[:,:,:c] + img1
            img_sa2 = img[:,:,c:] + img2

        img1 = img_sa1.reshape(batch, h, w, c).transpose(-1, 1)
        img2 = img_sa2.reshape(batch, h, w, c).transpose(-1, 1)

        return img1, img2
    

class Process_Encoder(nn.Module):
    """
    One visual transformer block
    """
    def __init__(self, n_layers, feature_size, heads, hidden_dim, attention_dim = 512, dropout = 0.):
        super(Process_Encoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        feature_dim=2048
        self.h_embedding = nn.Embedding(h_feat, int(channels/2))
        self.w_embedding = nn.Embedding(w_feat, int(channels/2))
        self.selftrans = nn.ModuleList([])
        for i in range(n_layers):                 
            self.selftrans.append(nn.ModuleList([
                Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False),
                Transformer(channels, channels, heads, attention_dim, hidden_dim, dropout, norm_first=False),
            ]))
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self._reset_parameters()
        self.Conv1 = nn.Conv2d(feature_dim*2, feature_dim, kernel_size = 1)
        self.LN = resblock(feature_dim, feature_dim)
        self.Conv2 = nn.Conv2d(1408*4, feature_dim, kernel_size = 1)
    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x1, x2,video):
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim = 1) + x_sam.unsqueeze(1) #(batch_size, 2channel, enc_image_size, enc_image_size)
        x = self.LN(self.Conv1(x))
        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(0,2,1)#(hw, batch_size, feature_dim)

        B,_,_=video.shape  #(batch,2*(1+2*256),1408)
        # print()
        video=video.reshape(B,2,-1,1408)[:,:,1:,:]
        video=video.reshape(B,2,-1,256,1408).permute(0,1,2,4,3).reshape(B,-1,16,16)
        video=self.Conv2(video).reshape(B,2048,-1).permute(0,2,1)
        for (l1,l2) in self.selftrans:        
            # print(video.shape,x.shape)   
            video = l1(video, x, x) + video
            x = l2(x, video, video) + x

        return x.permute(1,0,2)
    



class Renconstruct_img(nn.Module):
    """
    One visual transformer block
    """
    def __init__(self, re_size=32, in_channels=2048 ,out_channels=3):
        super(Renconstruct_img, self).__init__()
        self.upsampling = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, fea1,fea2,img1,img2):
        # print(fea1.shape,fea2.shape,img1.shape,img2.shape)
        img1_rec=self.upsampling(fea1)
        img2_rec=self.upsampling(fea2)
        img1=F.interpolate(img1, size=(32, 32),mode='bicubic', align_corners=False)
        img2=F.interpolate(img2, size=(32, 32),mode='bicubic', align_corners=False)
        
        return self.mse_loss(img1,img1_rec)+self.mse_loss(img2,img2_rec)
    