import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
# from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from util.function import normal,normal_style
from util.function import calc_mean_std
import scipy.stats as stats
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA
from PIL import Image
import model.transformer as trans



# from models.ViT_helper import to_2tuple, trunc_normal_

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(256,256), patch_size=(8,8), in_chans=3, embed_dim=512):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        return x


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
class StyTrans(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self,encoder,decoder,PatchEmbed, transformer,args):

        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model       
        self.decode = decoder
        self.embedding = PatchEmbed

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    def forward(self, samples_c: NestedTensor,samples_s: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """
        content_input = samples_c
        style_input = samples_s
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s) 
        
        # ### features used to calcate loss 

        content_feats = self.encode_with_intermediate(samples_c.tensors)

        style_feats = self.encode_with_intermediate(samples_s.tensors)
        # print(style_feats[0].shape)
        # pca = PCA(n_components=3)
        # feature_map=style_feats[0]-content_feats[0]
        
        # feature_map_flatten = feature_map.cpu().detach().numpy().transpose((0,2,3,1)).reshape(-1,64)  # 转换为numpy数组
        # # print(feature_map_flatten.shape)
        # feature_map_3d = pca.fit_transform(feature_map_flatten)
        # # print(feature_map_3d.shape)
        # # 归一化到0-1之间
        # feature_map_3d = (feature_map_3d - feature_map_3d.min(axis=0)) / (feature_map_3d.max(axis=0) - feature_map_3d.min(axis=0))*255
        
        # feature_map_3d=feature_map_3d.reshape(512,512,3)
        # # print(feature_map_3d.shape)
        # image = Image.fromarray(feature_map_3d.astype(np.uint8))

        # # 显示图像
        # image.save('out/test_000374_dif.png')
        
        # print(style_feats[-1].shape)
        ### Linear projection
        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)
        # print(style[0].shape)
        # # postional embedding is calculated in transformer.py
        # feature_map1 = content_feats[0].view(64,-1)
        # feature_map2 = style_feats[0].view(64,-1)
        
        # # 计算余弦相似度，结果是一个 [512, 64*64] 的张量
        # cosine_sim = cosine_similarity(feature_map1, feature_map2, dim=0)
        
        # # 将余弦相似度的形状调整回 [64, 64] 以便于可视化
        # cosine_sim = cosine_sim.view(512, 512)
        # cosine_sim=cosine_sim.detach().to('cpu').numpy()
        # # cosine_sim[cosine_sim >= 0.25] = np.nan
        # # 可视化余弦相似度
        # plt.imshow(cosine_sim, cmap='coolwarm_r')
        # plt.colorbar()
        # plt.title('Cosine Similarity of Pixel Features')
        # plt.savefig('out/test_000374_sim.png')
        # exit(0)

        pos_s = None
        pos_c = None

        mask = None
        hs = self.transformer(style, mask , content, pos_c, pos_s)   
        Ics = self.decode(hs)

        Ics_feats = self.encode_with_intermediate(Ics)
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1]))+self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))


        #ORIGIN style loss
        loss_ori_s = self.calc_style_loss(content_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_ori_s += self.calc_style_loss(content_feats[i], style_feats[i])
        

        # Style loss
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
            
        
        Icc = self.decode(self.transformer(content, mask , content, pos_c, pos_c))
        Iss = self.decode(self.transformer(style, mask , style, pos_s, pos_s))    

        #Identity losses lambda 1    
        loss_lambda1 = self.calc_content_loss(Icc,content_input)+self.calc_content_loss(Iss,style_input)
        
        #Identity losses lambda 2
        Icc_feats=self.encode_with_intermediate(Icc)
        Iss_feats=self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0])+self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i])+self.calc_content_loss(Iss_feats[i], style_feats[i])
        # Please select and comment out one of the following two sentences
        return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2,loss_ori_s   #train
        # return Ics    #test 







class StyTrans2(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self):

        super().__init__()

        self.transformer = trans.Transformer()
        self.embedding = PatchEmbed()

        from collections import OrderedDict


        new_state_dict = OrderedDict()
        state_dict = torch.load('/data/lky/proj/chang/styleTransfer/StyTR-2/experiments/transformer_iter_160000.pth')
        for k, v in state_dict.items():
            #namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        self.transformer.load_state_dict(new_state_dict)

        new_state_dict = OrderedDict()
        state_dict = torch.load('/data/lky/proj/chang/styleTransfer/StyTR-2/experiments/embedding_iter_160000.pth')
        for k, v in state_dict.items():
            #namekey = k[7:] # remove `module.`
            namekey = k
            new_state_dict[namekey] = v
        self.embedding.load_state_dict(new_state_dict)
        for param in self.embedding.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False
        self.conv1=nn.Conv2d(512,1024,kernel_size=3,stride = 2,padding=1)
        self.conv2=nn.Conv2d(1024,2048,kernel_size=3,stride = 2,padding=1)
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    def forward(self, samples_c: NestedTensor,samples_s: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """

        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s) 
        

        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)

        pos_s = None
        pos_c = None

        mask = None
        hs = self.transformer(style, mask , content, pos_c, pos_s)   
        
        # print(hs.shape)
        hs=self.conv1(hs)
        hs=self.conv2(hs)
        batchsize=hs.shape[0]
        
        # print(hs.shape)



        # Please select and comment out one of the following two sentences
        return hs.permute(0,2,3,1).reshape(batchsize,-1,2048)
        # return Ics    #test 