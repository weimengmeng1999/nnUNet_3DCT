#!/usr/bin/env python3
"""
Example usage of nnUNetPredictorWithFeatures class
This shows how to get both segmentation logits and encoder features
"""

import torch
import numpy as np
from predict_cls_new import nnUNetPredictorWithFeatures

def main():
    # Initialize the predictor
    predictor = nnUNetPredictorWithFeatures(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
        verbose=True,
        allow_tqdm=True
    )
    
    # Initialize from your trained model folder
    model_folder = "/path/to/your/trained/model"  # Replace with actual path
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),  # Use fold 0, or specify multiple folds
        checkpoint_name='checkpoint_final.pth'
    )
    
    # Create some dummy data (replace with your actual data)
    # Shape should be (channels, x, y, z) for 3D or (channels, x, y) for 2D
    dummy_data = torch.randn(1, 128, 128, 128)  # 3D example
    
    # Get both logits and features
    seg_logits, encoder_features = predictor.predict_logits_from_preprocessed_data(dummy_data)
    
    print(f"Segmentation logits shape: {seg_logits.shape}")
    print(f"Encoder features shape: {encoder_features.shape if encoder_features is not None else 'None'}")
    
    # Now you can use both outputs
    if encoder_features is not None:
        # Use encoder features for classification or other tasks
        print("Encoder features available!")
        # Example: Global average pooling for classification
        if len(encoder_features.shape) == 4:  # 2D
            global_features = torch.mean(encoder_features, dim=(2, 3))
        elif len(encoder_features.shape) == 5:  # 3D
            global_features = torch.mean(encoder_features, dim=(2, 3, 4))
        else:
            global_features = encoder_features
            
        print(f"Global features shape: {global_features.shape}")
    
    # Use segmentation logits as before
    seg_probs = torch.softmax(seg_logits, dim=0)
    print(f"Segmentation probabilities shape: {seg_probs.shape}")

if __name__ == "__main__":
    main() 