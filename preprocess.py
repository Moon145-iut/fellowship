import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class SlumDatasetPreprocessor:
    def __init__(self, data_dir='old', img_size=(256, 256)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.areas = []
        
    def load_dataset(self):
        """Load images and their corresponding JSON annotations"""
        print("Loading dataset...")
        for filename in tqdm(os.listdir(self.data_dir)):
            if filename.endswith('.jpg') and not '___fuse' in filename:
                
                img_path = os.path.join(self.data_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not load image: {img_path}")
                    continue
            
                json_path = os.path.join(self.data_dir, filename + '.json')
                if not os.path.exists(json_path):
                    print(f"No JSON file found for: {img_path}")
                    continue
                
                try:
                    with open(json_path, 'r') as f:
                        annotations = json.load(f)
                    
                    slum_annotations = []
                    for instance in annotations.get('instances', []):
                        if instance.get('className') == 'Slum':
                            slum_annotations.append(instance)

                    if slum_annotations:
                        
                        area = filename.split('-')[0] if '-' in filename else filename.split('_')[0]
                        
                        self.images.append(img)
                        self.labels.append(slum_annotations)
                        self.areas.append(area)
                    
                except Exception as e:
                    print(f"Error loading {json_path}: {str(e)}")
                    continue
        
        print(f"Loaded {len(self.images)} images")
    
    def preprocess_image(self, img):
        """Apply preprocessing steps to an image"""
    
        img = cv2.resize(img, self.img_size)
        
        
        img = img.astype(np.float32) / 255.0
        
    
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        

        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
    
        features = self.extract_features(img)
        
    
        enhanced_img = np.concatenate([
            img,
            features
        ], axis=-1)
        
        
        enhanced_img = (enhanced_img - enhanced_img.min()) / (enhanced_img.max() - enhanced_img.min())
        
        return enhanced_img
    
    def create_binary_mask(self, annotations):
        """Create binary mask for slum areas"""
        mask = np.zeros(self.img_size, dtype=np.float32)
        
        h, w = self.img_size
        orig_h, orig_w = None, None
        
        for ann in annotations:
            if isinstance(ann, dict) and 'points' in ann:
                points = np.array(ann['points'])
                
                # Skip if points array is empty
                if len(points) == 0:
                    continue
                
                if orig_h is None and 'imageHeight' in ann:
                    orig_h = ann['imageHeight']
                    orig_w = ann['imageWidth']
                    
                    # Calculate scale factors
                    scale_x = w / orig_w
                    scale_y = h / orig_h
                    
                    # Scale points
                    points[:, 0] = points[:, 0] * scale_x
                    points[:, 1] = points[:, 1] * scale_y
                
                # Ensure points are properly formatted for fillPoly
                points = points.reshape((-1, 1, 2))
                points = points.astype(np.int32)
                
                # Fill polygon
                cv2.fillPoly(mask, [points], 1)
        
        return mask
    
    def extract_features(self, img):
        """Extract features highlighting dense settlements and infrastructure"""
        # Convert to grayscale
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        # 1. Edge Detection for building boundaries
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = edges / np.max(edges)
        
        # 2. Local Binary Pattern for texture analysis
        def get_lbp(img, points=8, radius=1):
            n_points = points
            lbp = np.zeros_like(img)
            for i in range(radius, img.shape[0] - radius):
                for j in range(radius, img.shape[1] - radius):
                    center = img[i, j]
                    pattern = []
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = i + int(round(radius * np.cos(angle)))
                        y = j + int(round(radius * np.sin(angle)))
                        pattern.append(1 if img[x, y] >= center else 0)
                    lbp[i, j] = int(''.join(map(str, pattern)), 2)
            return lbp
        
        lbp = get_lbp(gray)
        lbp = lbp.astype(np.float32) / np.max(lbp)
        
        # 3. GLCM for density analysis
        def get_glcm_features(img):
            glcm = np.zeros((256, 256))
            rows, cols = img.shape
            for i in range(rows-1):
                for j in range(cols-1):
                    glcm[img[i,j], img[i,j+1]] += 1
            glcm = glcm / np.sum(glcm)
            return glcm
        
        glcm = get_glcm_features(gray)
        glcm_energy = np.sqrt(np.sum(glcm**2))
        
        # 4. Morphological operations for structure detection
        kernel = np.ones((5,5), np.uint8)
        morph = cv2.morphologyEx(edges.astype(np.float32), cv2.MORPH_CLOSE, kernel)
        
        # Combine features
        features = np.stack([
            edges,
            lbp,
            morph,
            np.full_like(edges, glcm_energy)
        ], axis=-1)
        
        # Normalize combined features
        features = (features - features.min()) / (features.max() - features.min())
        
        return features
    
    def split_dataset(self, test_size=0.2, val_size=0.2):
        """Split dataset into train, validation, and test sets"""
        # First split: train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            self.images, self.labels, test_size=test_size, random_state=42
        )
        
        # Second split: train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size/(1-test_size), random_state=42
        )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def visualize_samples(self, num_samples=5):
        """Visualize sample images with their features and masks"""
        if len(self.images) == 0:
            print("No images loaded. Please load dataset first.")
            return
            
        num_samples = min(num_samples, len(self.images))
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
        
        for i in range(num_samples):
            
            img = self.images[i]
            axes[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
        
            preprocessed = self.preprocess_image(img)
            features = self.extract_features(img)
            
           
            axes[i, 1].imshow(features[:,:,0], cmap='jet')
            axes[i, 1].set_title('Edge Features')
            axes[i, 1].axis('off')
            
            
            axes[i, 2].imshow(features[:,:,1], cmap='jet')
            axes[i, 2].set_title('Texture Features')
            axes[i, 2].axis('off')
            
            
            mask = self.create_binary_mask(self.labels[i])
            axes[i, 3].imshow(mask, cmap='gray')
            axes[i, 3].set_title('Slum Mask')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig('preprocessing_visualization.png')
        plt.close()

def main():

    preprocessor = SlumDatasetPreprocessor()
    
    preprocessor.load_dataset()
   
    preprocessor.visualize_samples()
   
    splits = preprocessor.split_dataset()
    print(f"Training set size: {len(splits['train'][0])}")
    print(f"Validation set size: {len(splits['val'][0])}")
    print(f"Test set size: {len(splits['test'][0])}")

if __name__ == "__main__":
    main() 