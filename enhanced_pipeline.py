import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import SlumDetector
from unsupervised import UnsupervisedValidator
from preprocess import SlumDatasetPreprocessor
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SlumDataset

class EnhancedSlumDetector:
    def __init__(self):
        self.supervised_model = SlumDetector()
        self.unsupervised_validator = UnsupervisedValidator()
        self.preprocessor = SlumDatasetPreprocessor()
        
    def train(self, num_epochs=50, batch_size=32, learning_rate=0.001):
        """Train the supervised model"""
        print("Training supervised model...")
        
        # Create dataloaders
        self.preprocessor.load_dataset()
        splits = self.preprocessor.split_dataset()
        
        # Create train and validation datasets
        train_dataset = SlumDataset(splits['train'][0], splits['train'][1], self.preprocessor)
        val_dataset = SlumDataset(splits['val'][0], splits['val'][1], self.preprocessor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Train the model
        self.supervised_model.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
    
    def predict_and_validate(self, image_path):
        """Predict slum areas and validate using unsupervised techniques"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get supervised model prediction
        preprocessed = self.preprocessor.preprocess_image(image)
        preprocessed_tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).unsqueeze(0)
        preprocessed_tensor = preprocessed_tensor.to(self.supervised_model.device)
        
        with torch.no_grad():
            model_pred = self.supervised_model.model(preprocessed_tensor)
            model_pred = model_pred.squeeze().cpu().numpy()
        
        # Validate using unsupervised techniques
        validation_results = self.unsupervised_validator.validate_results(image, model_pred)
        
        return {
            'model_prediction': model_pred,
            'validation_results': validation_results
        }
    
    def visualize_results(self, image_path, results):
        """Visualize both supervised and unsupervised results with enhanced heatmaps"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Create output directory if it doesn't exist
        os.makedirs('outputs/heatmaps', exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Resize image to match prediction size if needed
        image = cv2.resize(image, (results['model_prediction'].shape[1], results['model_prediction'].shape[0]))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        # 1. Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')
        
        # 2. Supervised model prediction heatmap
        prediction = results['model_prediction']
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap((prediction * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        axes[0, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Prediction Heatmap Overlay', fontsize=14)
        axes[0, 1].axis('off')
        
     
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.imshow(prediction, cmap='jet', alpha=0.5)
        plt.colorbar(label='Confidence Score')
        plt.title(f'Slum Detection Heatmap - {base_name}')
        plt.axis('off')
        plt.savefig(f'outputs/heatmaps/{base_name}_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        cluster_vis = results['validation_results']['visualization']
        axes[1, 0].imshow(cv2.cvtColor(cluster_vis, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Unsupervised Clustering', fontsize=14)
        axes[1, 0].axis('off')
        
        combined = 0.7 * results['model_prediction'] + \
                  0.3 * results['validation_results']['cluster_mask']
        combined = (combined - combined.min()) / (combined.max() - combined.min())
        
        combined_heatmap = cv2.applyColorMap((combined * 255).astype(np.uint8), cv2.COLORMAP_JET)
        combined_overlay = cv2.addWeighted(image, 0.7, combined_heatmap, 0.3, 0)
        axes[1, 1].imshow(cv2.cvtColor(combined_overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Combined Confidence Overlay', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.suptitle(f'Slum Detection Analysis - {base_name}', fontsize=16, y=0.95)
        plt.tight_layout()
        
        save_path = f'outputs/test_predictions/{base_name}_full_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        metrics_path = f'outputs/test_predictions/{base_name}_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write(f"Analysis Results for {base_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"IoU Score: {results['validation_results']['iou']:.4f}\n\n")
            f.write("Cluster Statistics:\n")
            f.write("-" * 30 + "\n")
            for cluster_id, stats in results['validation_results']['cluster_stats'].items():
                f.write(f"\nCluster {cluster_id}:\n")
                f.write(f"  Size: {stats['size']}\n")
                f.write(f"  Edge Density: {stats['edge_density']:.4f}\n")
                f.write(f"  Color Variation: {stats['color_variation']:.4f}\n")
                f.write(f"  Structural Complexity: {stats['structural_complexity']:.4f}\n")
                f.write(f"  Slum Score: {results['validation_results']['slum_scores'][cluster_id]:.4f}\n")

def main():
   
    detector = EnhancedSlumDetector()
    
    
    detector.train()
    
    # Process test images
    test_dir = 'old'  
    for image_name in tqdm(os.listdir(test_dir)):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(test_dir, image_name)
            try:
                
                results = detector.predict_and_validate(image_path)
                
                
                detector.visualize_results(image_path, results)
                
            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")
                continue

if __name__ == "__main__":
    main() 