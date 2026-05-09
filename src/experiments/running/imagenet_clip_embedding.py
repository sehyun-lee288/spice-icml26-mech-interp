
# from dsets import get_dataset
# from transformers import AutoProcessor, CLIPVisionModel
# import torch
# from tqdm import tqdm

# model_CLIP = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
# model_CLIP.eval().cuda();
# processor_CLIP = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)


# dataset_name = 'imagenet'
# data_path = '/project/data/external/ILSVRC/Data/CLS-LOC'

# from torchvision.datasets import ImageFolder
# from torchvision import transforms

# class ProcessorCLIPImageFolder(ImageFolder):
#     def __init__(self, root, processor, **kwargs):
#         super().__init__(root, **kwargs)
#         self.processor = processor

#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         # Returns just the processed image tensor, not the target.
#         processed = self.processor(images=sample, return_tensors="pt")['pixel_values'].squeeze(0)
#         return processed

# val_dir = f"{data_path}/val"
# imagenet_clip_dataset = ProcessorCLIPImageFolder(val_dir, processor_CLIP)

# dataloader = torch.utils.data.DataLoader(imagenet_clip_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

# clip_embeddings = []
# for imgs in tqdm(dataloader, desc="Extracting CLIP embeddings"):
#     imgs = imgs.cuda()
#     clip_embeddings.append(model_CLIP(imgs).last_hidden_state.mean(1).detach().cpu())
# clip_embeddings = torch.cat(clip_embeddings)

# torch.save(clip_embeddings, "/project/results/orig_CLIP_embedding/clip_embeddings_imagenet_val.pt")


import torch
import clip
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torchvision.datasets import ImageFolder
from torchvision import transforms 
from tqdm import tqdm

def tensor_to_pil(tensor_image):
    return to_pil_image(tensor_image)

dataset_name = 'imagenet'
data_path = '/project/data/external/ILSVRC/Data/CLS-LOC'


dataset = ImageFolder(
    root=f"{data_path}/val",
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

device = 'cuda'
clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, jit=False) 

embeddings_list = []
for batch_images, _ in tqdm(dataloader, total=len(dataloader)):
    batch_pil = [tensor_to_pil(img) for img in batch_images]
    batch_preprocessed = torch.stack([clip_preprocess(img) for img in batch_pil]).to(device)
    batch_embeddings = clip_model.encode_image(batch_preprocessed)
    batch_embeddings = F.normalize(batch_embeddings, dim=1)
    embeddings_list.append(batch_embeddings.detach().cpu().numpy())

embeddings_array = np.concatenate(embeddings_list, axis=0)
np.save("/project/results/orig_CLIP_embedding/github_clip_embeddings_imagenet_val.npy", embeddings_array)

# embeddings = np.vstack(embeddings_list)
# similarity_matrix = np.dot(embeddings, embeddings.T)