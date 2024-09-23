import os
import numpy as np
from urllib.request import urlopen
import torchvision.transforms as transforms  
import folder_paths
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image,ImageOps, ImageFilter
import torch.nn as nn
import torch

# comfy_path = os.path.dirname(folder_paths.__file__)
# custom_nodes_path = os.path.join(comfy_path, "custom_nodes")


# 指定本地分割模型文件夹的路径
model_folder_path = os.path.join(folder_paths.models_dir,"segformer_b2_sleeves")

processor = SegformerImageProcessor.from_pretrained(model_folder_path)
model = AutoModelForSemanticSegmentation.from_pretrained(model_folder_path)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# 切割服装
def get_segmentation(tensor_image):
    cloth = tensor2pil(tensor_image)
    # 预处理和预测
    inputs = processor(images=cloth, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=cloth.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    return pred_seg,cloth


class segformer_b2_sleeves:
   
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {     
                 "image":("IMAGE", {"default": "","multiline": False}),
                 "Background": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                 "Upper_torso": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                 "Left_pants": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                 "Right_pants": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                 "Skirts": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                 "Left_sleeve": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                 "Right_sleeve": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                 "Outer_collar": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                 "Inner_collar": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask_image",)
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "CXH"

    def sample(self,image,Background,Upper_torso,Left_pants,Right_pants,Skirts,Left_sleeve,Right_sleeve,Outer_collar,Inner_collar):
        
        results = []
        for item in image:
        
            # seg切割结果，衣服pil
            pred_seg,cloth = get_segmentation(item)
            labels_to_keep = [0]
            # if not Background:
            #     labels_to_keep.append(0)
            if not Upper_torso:
                labels_to_keep.append(1)
            if not Left_pants:
                labels_to_keep.append(2)
            if not Right_pants:
                labels_to_keep.append(3)
            if not Skirts:
                labels_to_keep.append(4)
            if not Left_sleeve:
                labels_to_keep.append(5)
            if not Right_sleeve:
                labels_to_keep.append(6)
            if not Outer_collar:
                labels_to_keep.append(7)
            if not Inner_collar:
                labels_to_keep.append(8)

            print(labels_to_keep)
                
            mask = np.isin(pred_seg, labels_to_keep).astype(np.uint8)
            
            # 创建agnostic-mask图像
            mask_image = Image.fromarray(mask * 255)
            mask_image = mask_image.convert("RGB")
            mask_image = pil2tensor(mask_image)
            results.append(mask_image)

        return (torch.cat(results, dim=0),)