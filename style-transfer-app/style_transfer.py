import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess images
loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    lambda x: x[:3, :, :],  # remove alpha if present
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
unloader = transforms.Compose([
    transforms.Normalize(mean=[-2.12, -2.04, -1.80],
                         std=[4.37, 4.46, 4.44]),
    transforms.ToPILImage()
])