import torch
import os
from torch import nn
import loralib as lora
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
from torchvision import transforms
decord.bridge.set_bridge("torch")
import sys
your_library_path = '/root/video_model'
# 添加到sys.path
sys.path.append(your_library_path)


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)
    # print('frame_indices',frame_indices.shape)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])
    
    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)
    
    if padding:
        frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    # print(frames.shape)
    T_, C, H, W = frames.shape
    
    sub_img = frames.reshape(
        1, T_, 3, H//resolution, resolution, W//resolution, resolution
    ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()

    glb_img = F.interpolate(
        frames.float(), size=(resolution, resolution), mode='bicubic', align_corners=False
    ).to(sub_img.dtype).unsqueeze(0)
    
    frames = torch.cat([sub_img, glb_img]).unsqueeze(0)
    # print("haha",frames.shape)
    # frames = glb_img.unsqueeze(0)
    # print(frames.shape)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames

def HD_transform_padding(frames, image_size=224, hd_num=6):
    def _padding_224(frames):
        _, _, H, W = frames.shape
        tar = int(np.ceil(H / 224) * 224)
        top_padding = (tar - H) // 2
        bottom_padding = tar - H - top_padding
        left_padding = 0
        right_padding = 0

        padded_frames = F.pad(
            frames,
            pad=[left_padding, right_padding, top_padding, bottom_padding],
            mode='constant', value=255
        )
        return padded_frames

    _, _, H, W = frames.shape
    trans = False
    if W < H:
        frames = frames.flip(-2, -1)
        trans = True
        width, height = H, W
    else:
        width, height = W, H

    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * image_size)
    new_h = int(new_w / ratio)

    resized_frames = F.interpolate(
        frames, size=(new_h, new_w),
        mode='bicubic',
        align_corners=False
    )
    padded_frames = _padding_224(resized_frames)

    if trans:
        padded_frames = padded_frames.flip(-2, -1)

    return padded_frames

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio


def HD_transform_no_padding(frames, image_size=224, hd_num=6, fix_ratio=(1,1)):
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    # calculate the existing video aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    if fix_ratio:
        target_aspect_ratio = fix_ratio
    else:
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the frames
    resized_frame = F.interpolate(
        frames, size=(target_height, target_width),
        mode='bicubic', align_corners=False
    )
    return resized_frame

# 定义一个函数来应用 LoRA 到 FC 层
def apply_lora_to_fc(layer, rank=4,bias=True,if_qkv=False):
    # 假设 layer 是一个 nn.Linear 层
    # 你可以根据需要调整 rank 的值
    if if_qkv:
        lora_layer = lora.Linear(layer.in_features, 4224, r=rank,bias=bias)
    else:
        lora_layer = lora.Linear(layer.in_features, layer.out_features, r=rank,bias=bias)
    lora_layer.weight.data = layer.weight.data.clone()
    if bias:
        lora_layer.bias.data = layer.bias.data.clone()
    # print(lora_layer)
    return lora_layer.to(torch.bfloat16)

def apply_lora_to_model(model, layer_indices, rank=4):
    for idx in layer_indices:
        for param in model.blocks[idx].parameters():
                param.requires_grad = False
        block = model.blocks[idx]
        block.mlp.fc1 = apply_lora_to_fc(block.mlp.fc1, rank)
        block.mlp.fc2 = apply_lora_to_fc(block.mlp.fc2, rank)
        block.attn.proj = apply_lora_to_fc(block.attn.proj, rank)
        # block.attn.qkv.requires_grad = True
        # print('qkv')
        # block.attn.qkv = apply_lora_to_fc(block.attn.proj, rank,bias=False,if_qkv=True)
        

class Video_encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, vision_encoder_pretrain='checkpoints/vision_encoder.pth', if_lora=True):
        super(Video_encoder, self).__init__()
        self.model = torch.load(vision_encoder_pretrain)
        for param in self.model.parameters():
            param.requires_grad = False
        if if_lora:
            layer_indices = [38]  
            rank = 4  
            apply_lora_to_model(self.model, layer_indices, rank=rank)
        else:
            for param in self.model.blocks[38].parameters():
                param.requires_grad = True
        self.att_liner=nn.Linear(1408,2048)

    def forward(self,video_tensor):
        if len(video_tensor.shape) == 5:
            B, T, C, H, W = video_tensor.shape
            N = 1
        else:
            B, N, T, C, H, W = video_tensor.shape
            batch_size=B
            
        video_tensor = video_tensor.reshape(B*N, T, C, H, W).permute(0, 2, 1, 3, 4)
        B = video_tensor.shape[0]
        
        T = video_tensor.shape[2]
        use_image = True if T == 1 else False
        
        image_embeds = self.model(video_tensor, use_image=use_image)
       
        C = image_embeds.shape[-1]
        image_embeds = image_embeds.reshape(batch_size, -1, C)
        

        return self.att_liner(image_embeds.float()),image_embeds.float()
                
    
class Sty_fusion(nn.Module):
    """
    Encoder.
    """

    def __init__(self):
        super(Sty_fusion, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self,video_tensor,mask):
        mask=mask.view(-1,256)
        mask=torch.cat((mask,mask),dim=1)
        
        mask = torch.nn.functional.pad(mask, (1, 0), 'constant', 0)
        mask=torch.cat((mask,mask),dim=1).unsqueeze(2)

        return video_tensor*mask#(1-self.alpha)*video_tensor*mask+self.alpha*video_tensor
             

class Clip_fusion(nn.Module):
    """
    Encoder.
    """

    def __init__(self):
        super(Clip_fusion, self).__init__()
        self.att_liner=nn.Linear(1024,2048)
    def forward(self,imA,imB):
        return self.att_liner(torch.cat((imA,imB),dim=1))#(1-self.alpha)*video_tensor*mask+self.alpha*video_tensor
             