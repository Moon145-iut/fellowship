import torch
from torch.utils.data import Dataset
import numpy as np

class SlumDataset(Dataset):
    """Dataset class for slum detection"""
    def __init__(self, images, labels, preprocessor):
        self.images = images
        self.labels = labels
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
      
        image = self.images[idx]
        label = self.labels[idx]
        
        
        processed_image = self.preprocessor.preprocess_image(image)
      
        mask = self.preprocessor.create_binary_mask(label)
        

        image_tensor = torch.from_numpy(processed_image).float()
        mask_tensor = torch.from_numpy(mask).float()
        
    
        image_tensor = image_tensor.permute(2, 0, 1) 
        
        return image_tensor, mask_tensor 