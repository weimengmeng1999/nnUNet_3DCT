import torch
from torch import nn
from typing import Tuple, Union, List
# for alternative nnunet structure

class nnUNetWithFeatures(nn.Module):
    """Wrapper around nnUNet that also returns encoder features"""
    
    def __init__(self, original_network):
        super().__init__()
        self.original_network = original_network
        
    def forward(self, x):
        """
        Forward pass that returns both segmentation outputs and encoder features.
        This is a more robust approach that directly accesses the network's internal structure.
        """
        
        # Debug: Print network attributes to understand structure
        # print(f"Network type: {type(self.original_network)}")
        # print(f"Has conv_blocks_context: {hasattr(self.original_network, 'conv_blocks_context')}")
        # print(f"Has encoder: {hasattr(self.original_network, 'encoder')}")
        # print(f"Has decoder: {hasattr(self.original_network, 'decoder')}")
        
        # For standard nnUNet architecture with conv_blocks_context
        if hasattr(self.original_network, 'conv_blocks_context'):
            # print("Using _forward_standard_nnunet")
            return self._forward_standard_nnunet(x)
        # For newer architectures with explicit encoder/decoder
        elif hasattr(self.original_network, 'encoder') and hasattr(self.original_network, 'decoder'):
            # print("Using _forward_explicit_encoder_decoder")
            return self._forward_explicit_encoder_decoder(x)
        # For other architectures, try to use hooks as fallback
        else:
            # print("Using _forward_with_hooks")
            return self._forward_with_hooks(x)

        # return self._forward_standard_nnunet(x)
    
    def _forward_standard_nnunet(self, x):
        """Forward pass for standard nnUNet architecture"""
        skips = []
        features = x
        
        # Forward through encoder stages
        for i, block in enumerate(self.original_network.conv_blocks_context):
            features = block(features)
            if i < len(self.original_network.conv_blocks_context) - 1:
                skips.append(features)
                features = self.original_network.downsample_layers[i](features)
        
        # Store bottleneck features for classification (this is what we want!)
        encoder_features = features
        
        # Forward through decoder stages
        features = self.original_network.conv_blocks_localization[0](features)
        for i in range(len(skips)):
            features = self.original_network.upsample_layers[i](features)
            features = torch.cat([features, skips[-(i+1)]], dim=1)
            features = self.original_network.conv_blocks_localization[i+1](features)
        
        # Apply segmentation heads
        seg_outputs = []
        if hasattr(self.original_network, 'seg_outputs') and isinstance(self.original_network.seg_outputs, nn.ModuleList):
            for head in self.original_network.seg_outputs:
                seg_outputs.append(head(features))
        else:
            # Single output head
            seg_outputs.append(self.original_network.seg_outputs(features))
        
        return seg_outputs, encoder_features
    
    def _forward_explicit_encoder_decoder(self, x):
        """Forward pass for newer architectures with explicit encoder/decoder"""
        # Use the original network's forward pass to get segmentation outputs
        # This ensures all skip connections are handled correctly
        seg_outputs = self.original_network(x)
        
        # Extract encoder features from the forward pass using hooks or by accessing intermediate activations
        # For now, we'll use a simple approach: get encoder features without gradients
        with torch.no_grad():
            encoder_features = self.original_network.encoder(x)
            
            # Get the bottleneck (deepest encoder features)
            if isinstance(encoder_features, (list, tuple)):
                encoder_bottleneck = encoder_features[-1]  # Last encoder stage
            else:
                encoder_bottleneck = encoder_features
        
        return seg_outputs, encoder_bottleneck

    
    def _forward_with_hooks(self, x):
        """Fallback forward pass using hooks"""
        # This is a fallback method that tries to use hooks
        # It's less reliable but might work for some architectures
        
        encoder_features = None
        
        def hook_fn(module, input, output):
            nonlocal encoder_features
            # Try to identify the encoder bottleneck
            if hasattr(module, 'conv_blocks_context') and len(module.conv_blocks_context) > 0:
                # This looks like the main UNet body
                if hasattr(module, 'conv_blocks_context'):
                    # Get the last encoder block output
                    encoder_features = module.conv_blocks_context[-1](input[0])
            elif hasattr(module, 'encoder'):
                # Newer architecture with explicit encoder
                encoder_features = module.encoder(input[0])
                if isinstance(encoder_features, (list, tuple)):
                    encoder_features = encoder_features[-1]  # Last encoder stage
        
        # Register the hook
        hook_handle = self.original_network.register_forward_hook(hook_fn)
        
        try:
            # Run the original forward pass
            seg_outputs = self.original_network.forward(x)
            
            # If we couldn't capture encoder features, create dummy features
            if encoder_features is None:
                batch_size = x.shape[0]
                spatial_shape = x.shape[2:]
                encoder_features = torch.randn(batch_size, 320, *spatial_shape, device=x.device)
                print("Warning: Could not extract encoder features using hooks, using dummy tensor")
            
        finally:
            # Remove the hook
            hook_handle.remove()
        
        return seg_outputs, encoder_features
    
    def __getattr__(self, name):
        """Delegate attribute access to original network"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_network, name)
    
    def state_dict(self, *args, **kwargs):
        """Delegate state_dict to original network"""
        return self.original_network.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        """Delegate load_state_dict to original network"""
        return self.original_network.load_state_dict(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        """Delegate parameters to original network"""
        return self.original_network.parameters(*args, **kwargs)
    
    def named_parameters(self, *args, **kwargs):
        """Delegate named_parameters to original network"""
        return self.original_network.named_parameters(*args, **kwargs) 