import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

class UnsupervisedValidator:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
    def extract_features(self, image):
        """Extract relevant features for clustering"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features
        glcm = self._calculate_glcm(gray)
        
        # Calculate edge features
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)
        
        # Calculate color features
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv, axis=(0, 1))
        
        # Calculate structural features
        structural_features = self._calculate_structural_features(gray)
        
        # Combine all features
        features = np.concatenate([
            glcm.flatten(),
            [edge_density],
            color_std,
            structural_features
        ])
        
        return features
    
    def _calculate_glcm(self, gray):
        """Calculate Gray Level Co-occurrence Matrix features"""
        glcm = np.zeros((256, 256))
        rows, cols = gray.shape
        
        for i in range(rows-1):
            for j in range(cols-1):
                glcm[gray[i,j], gray[i,j+1]] += 1
                glcm[gray[i,j], gray[i+1,j]] += 1
        
        # Normalize
        glcm = glcm / glcm.sum()
        return glcm
    
    def _calculate_structural_features(self, gray):
        """Calculate structural features related to building patterns"""
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate edge orientation histogram
        orientation = np.arctan2(sobely, sobelx)
        orientation_hist = np.histogram(orientation, bins=8, range=(-np.pi, np.pi))[0]
        
        # Calculate edge magnitude statistics
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude_stats = [
            np.mean(magnitude),
            np.std(magnitude),
            stats.skew(magnitude.flatten()),
            stats.kurtosis(magnitude.flatten())
        ]
        
        return np.concatenate([orientation_hist, magnitude_stats])
    
    def analyze_clusters(self, features):
        """Analyze clusters to identify potential slum areas"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Apply K-means clustering
        kmeans_labels = self.kmeans.fit_predict(scaled_features)
        
        # Apply DBSCAN for density-based clustering
        dbscan_labels = self.dbscan.fit_predict(scaled_features)
        
        # Calculate cluster characteristics
        cluster_stats = {}
        for i in range(self.n_clusters):
            cluster_mask = kmeans_labels == i
            cluster_features = features[cluster_mask]
            
            # Calculate statistics for each cluster
            stats_dict = {
                'size': np.sum(cluster_mask),
                'edge_density': np.mean(cluster_features[:, -1]),
                'color_variation': np.mean(cluster_features[:, -2:-1]),
                'structural_complexity': np.mean(cluster_features[:, -3:-2])
            }
            cluster_stats[i] = stats_dict
        
        return kmeans_labels, dbscan_labels, cluster_stats
    
    def identify_slum_clusters(self, cluster_stats):
        """Identify which clusters likely represent slum areas"""
        slum_scores = {}
        
        for cluster_id, stats in cluster_stats.items():
            # Calculate slum score based on characteristics
            score = (
                0.4 * stats['edge_density'] +  # High edge density indicates dense settlements
                0.3 * stats['color_variation'] +  # High color variation indicates informal structures
                0.3 * stats['structural_complexity']  # High structural complexity indicates irregular patterns
            )
            slum_scores[cluster_id] = score
        
        # Identify clusters with high slum scores
        slum_threshold = np.mean(list(slum_scores.values())) + np.std(list(slum_scores.values()))
        slum_clusters = [cluster_id for cluster_id, score in slum_scores.items() 
                        if score > slum_threshold]
        
        return slum_clusters, slum_scores
    
    def visualize_clusters(self, image, kmeans_labels, slum_clusters):
        """Visualize clustering results"""
        # Create colored mask for clusters
        height, width = image.shape[:2]
        cluster_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color slum clusters in red, others in blue
        for cluster_id in range(self.n_clusters):
            mask = kmeans_labels.reshape(height, width) == cluster_id
            if cluster_id in slum_clusters:
                cluster_mask[mask] = [0, 0, 255]  # Red for slum areas
            else:
                cluster_mask[mask] = [255, 0, 0]  # Blue for non-slum areas
        
        # Blend with original image
        alpha = 0.5
        overlay = cv2.addWeighted(image, 1-alpha, cluster_mask, alpha, 0)
        
        return overlay
    
    def validate_results(self, image, model_predictions):
        """Validate model predictions using unsupervised techniques"""
        
        features = self.extract_features(image)
        
        
        kmeans_labels, dbscan_labels, cluster_stats = self.analyze_clusters(features)
        
       
        slum_clusters, slum_scores = self.identify_slum_clusters(cluster_stats)
        
        
        visualization = self.visualize_clusters(image, kmeans_labels, slum_clusters)

        model_mask = (model_predictions > 0.5).astype(np.uint8)
        cluster_mask = np.zeros_like(model_mask)
        for cluster_id in slum_clusters:
            cluster_mask[kmeans_labels.reshape(model_mask.shape) == cluster_id] = 1
        
        # Calculate IoU
        intersection = np.sum(np.logical_and(model_mask, cluster_mask))
        union = np.sum(np.logical_or(model_mask, cluster_mask))
        iou = intersection / (union + 1e-6)
        
        return {
            'iou': iou,
            'visualization': visualization,
            'cluster_stats': cluster_stats,
            'slum_scores': slum_scores
        } 