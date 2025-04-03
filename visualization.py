import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import folium
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
from model import SlumDetector

class GradCAM:
    """
    Grad-CAM implementation for visualizing where the model is looking
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_image, target_class=None):
        # Forward pass
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot_output, retain_graph=True)
        
        # Get weights
        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weight the activations
        weighted_activations = (self.activations * gradients).sum(dim=1, keepdim=True)
        
        # Generate heatmap
        heatmap = F.relu(weighted_activations).squeeze()
        heatmap = heatmap.cpu().numpy()
        
        # Normalize
        heatmap = cv2.resize(heatmap, (input_image.shape[3], input_image.shape[2]))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap

class SlumVisualizer:
    """
    Class for visualizing slum detection results
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.grad_cam = GradCAM(model, target_layer)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def generate_visualization(self, image_path, output_path=None):
        """
        Generate comprehensive visualization with heatmap overlay and bounding boxes
        """
        # Load original image without any preprocessing
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = original_image.shape[:2]

        # Load and preprocess image for model input
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Get model prediction and Grad-CAM heatmap
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = prediction.squeeze().cpu().numpy()
        heatmap = self.grad_cam.generate_heatmap(input_tensor)
        
        # Create visualization figure
        plt.figure(figsize=(20, 10))
        
        # Original image (unprocessed)
        plt.subplot(231)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Raw prediction (resize to match original image size)
        plt.subplot(232)
        prediction_resized = cv2.resize(prediction, (orig_width, orig_height))
        plt.imshow(prediction_resized, cmap='jet')
        plt.title('Raw Prediction')
        plt.colorbar()
        plt.axis('off')
        
        # Grad-CAM heatmap (resize to match original image size)
        plt.subplot(233)
        heatmap_resized = cv2.resize(heatmap, (orig_width, orig_height))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        plt.imshow(heatmap_colored)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        # Overlay heatmap on original image
        alpha = 0.5
        overlay = cv2.addWeighted(original_image, 1-alpha, heatmap_colored, alpha, 0)
        plt.subplot(234)
        plt.imshow(overlay)
        plt.title('Heatmap Overlay')
        plt.axis('off')
        
        # Binary prediction with bounding boxes on original image
        plt.subplot(235)
        binary_mask = (prediction_resized > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = original_image.copy()
        for contour in contours:
            # Calculate area relative to original image size
            area = cv2.contourArea(contour)
            min_area = (orig_width * orig_height) * 0.001  # 0.1% of image size
            if area > min_area:  # Filter small regions
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Add confidence score using original image coordinates
                conf = np.mean(prediction_resized[y:y+h, x:x+w])
                cv2.putText(result, f'{conf:.2f}', (x, y-5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        plt.imshow(result)
        plt.title('Detected Regions')
        plt.axis('off')
        
        # Confidence histogram
        plt.subplot(236)
        plt.hist(prediction_resized.flatten(), bins=50, range=(0, 1))
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
        
        return result, prediction_resized, heatmap_resized
    
    def create_map_visualization(self, image_locations, predictions, output_path=None):
        """
        Create an interactive map with detailed markers for slum locations
        """
        # Create base map centered on the mean coordinates
        mean_lat = np.mean([loc[0] for loc in image_locations])
        mean_lon = np.mean([loc[1] for loc in image_locations])
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)
        
        # Add heatmap layer
        heat_data = []
        for (lat, lon), pred in zip(image_locations, predictions):
            if pred > 0.5:  # Only include high confidence predictions
                weight = float(pred)  # Convert prediction to weight
                heat_data.append([lat, lon, weight])
        
        folium.plugins.HeatMap(heat_data).add_to(m)
        
        # Add markers for each location
        for (lat, lon), pred in zip(image_locations, predictions):
            if pred > 0.5:  # High confidence prediction
                color = 'red'
                icon = 'warning'
                popup_text = f'High Risk Area\nConfidence: {pred:.2f}'
            elif pred > 0.3:  # Medium confidence
                color = 'orange'
                icon = 'info-sign'
                popup_text = f'Medium Risk Area\nConfidence: {pred:.2f}'
            else:  # Low confidence
                color = 'green'
                icon = 'ok-sign'
                popup_text = f'Low Risk Area\nConfidence: {pred:.2f}'
            
            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)
            
            # Add circle to show prediction confidence
            folium.Circle(
                location=[lat, lon],
                radius=100,  # meters
                color=color,
                fill=True,
                fillOpacity=pred,
                popup=f'Confidence: {pred:.2f}'
            ).add_to(m)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
        <p><i class="fa fa-warning" style="color: red;"></i> High Risk (>0.5)</p>
        <p><i class="fa fa-info-sign" style="color: orange;"></i> Medium Risk (0.3-0.5)</p>
        <p><i class="fa fa-ok-sign" style="color: green;"></i> Low Risk (<0.3)</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if output_path:
            m.save(output_path)
        
        return m

def main():
    # Initialize model and visualizer
    model = SlumDetector()
    model.load_weights('model_weights.pth')  # Use the correct method to load weights
    model.eval()
    
    # Get the target layer for Grad-CAM (adjust based on your model architecture)
    target_layer = model.backbone.layer4[-1]
    
    # Create visualizer
    visualizer = SlumVisualizer(model, target_layer)
    
    # Generate visualization for a single image
    visualizer.generate_visualization('example_image.jpg', 'visualization_output.png')
    
    # Create map visualization
    image_locations = [
        (23.7957, 90.3659),  # Example coordinates
        (23.8000, 90.3700)
    ]
    predictions = [0.8, 0.3]  # Example predictions
    visualizer.create_map_visualization(
        image_locations, 
        predictions, 
        'map_visualization.html'
    )

if __name__ == '__main__':
    main() 