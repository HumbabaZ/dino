#Combining DINO self-supervised learning with Twin Network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import vision_transformer as vits

#Architecture:
#- Two identical DINO ViT encoders (shared weights)
#- Feature concatenation
#- Fully connected classifier
class DINOTwinNetwork(nn.Module):
    
    def __init__(
        self,
        arch: str = 'vit_small',
        patch_size: int = 8,
        pretrained_weights: Optional[str] = None,
        embedding_dimension: int = 128,
        num_classes: int = 1,  # Binary classification for twin network
        freeze_backbone: bool = False,
        dropout_rate: float = 0.1
    ):
        super(DINOTwinNetwork, self).__init__()
        
        self.arch = arch
        self.patch_size = patch_size
        self.embedding_dimension = embedding_dimension
        
        # Build DINO backbone
        self.backbone = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0  # Remove classification head
        )
        
        # Load pre-trained DINO weights
        if pretrained_weights:
            self.load_dino_weights(pretrained_weights)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get embedding dimension from backbone
        backbone_dim = self.backbone.embed_dim
        
        # Feature projection head (optional)
        self.feature_projector = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Twin network classifier
        # Input: concatenated features from two images (2 * embedding_dimension)
        self.classifier = nn.Sequential(
            nn.Linear(2 * embedding_dimension, embedding_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dimension, embedding_dimension // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dimension // 2, num_classes),
            nn.Sigmoid()  # For binary classification
        )
        
        # Store parameters for saving/loading
        self.kwargs = {
            'arch': arch,
            'patch_size': patch_size,
            'embedding_dimension': embedding_dimension,
            'num_classes': num_classes,
            'freeze_backbone': freeze_backbone,
            'dropout_rate': dropout_rate
        }
    
    #Load DINO pre-trained weights
    def load_dino_weights(self, pretrained_weights_path: str):
        
        try:
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            
            # Handle different checkpoint formats
            if 'teacher' in state_dict:
                state_dict = state_dict['teacher']
            else:
                print("No teacher in checkpoint")
            
            # Remove prefixes if they exist
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            
            # Filter out head weights (we only want backbone)
            backbone_state_dict = {k: v for k, v in state_dict.items() 
                                 if not k.startswith('head')}
            
            msg = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            print(f'DINO weights loaded from {pretrained_weights_path}')
            print(f'Loading message: {msg}')
            
        except Exception as e:
            print(f'Error loading DINO weights: {e}')
            print('Continuing with random initialization...')
    
    #Encode a single image using DINO backbone
    def encode_single(self, x):
        # x shape: (batch_size, channels, height, width)
        features = self.backbone(x)  # (batch_size, embed_dim)
        projected_features = self.feature_projector(features)  # (batch_size, embedding_dimension)
        return projected_features
    
    #Forward pass for twin network            
    #Returns: similarity_scores: (batch_size, num_classes)
    def forward(self, x1, x2):
        
        # Encode both images
        features1 = self.encode_single(x1)  # (batch_size, embedding_dimension)
        features2 = self.encode_single(x2)  # (batch_size, embedding_dimension)
        
        # Concatenate features
        combined_features = torch.cat([features1, features2], dim=1)  # (batch_size, 2 * embedding_dimension)
        
        # Classify
        similarity_scores = self.classifier(combined_features)  # (batch_size, num_classes)
        
        return similarity_scores
    
    #Get feature embeddings for both images
    def get_embeddings(self, x1, x2):
        with torch.no_grad():
            features1 = self.encode_single(x1)
            features2 = self.encode_single(x2)
        return features1, features2


class DINOContrastiveNetwork(nn.Module):
    """
    Alternative implementation using contrastive/distance-based approach
    """
    
    def __init__(
        self,
        arch: str = 'vit_small',
        patch_size: int = 8,
        pretrained_weights: Optional[str] = None,
        embedding_dimension: int = 128,
        freeze_backbone: bool = False
    ):
        super(DINOContrastiveNetwork, self).__init__()
        
        # Build DINO backbone
        self.backbone = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0
        )
        
        # Load pre-trained weights
        if pretrained_weights:
            self.load_dino_weights(pretrained_weights)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        backbone_dim = self.backbone.embed_dim
        
        # Embedding projection
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, embedding_dimension)
        )
        
        self.kwargs = {
            'arch': arch,
            'patch_size': patch_size,
            'embedding_dimension': embedding_dimension,
            'freeze_backbone': freeze_backbone
        }
    
    def load_dino_weights(self, pretrained_weights_path: str):
        """Same as above"""
        try:
            state_dict = torch.load(pretrained_weights_path, map_location="cpu")
            
            if 'teacher' in state_dict:
                state_dict = state_dict['teacher']
            elif 'student' in state_dict:
                state_dict = state_dict['student']
            
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            
            backbone_state_dict = {k: v for k, v in state_dict.items() 
                                 if not k.startswith('head')}
            
            msg = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            print(f'DINO weights loaded from {pretrained_weights_path}')
            
        except Exception as e:
            print(f'Error loading DINO weights: {e}')
    
    def forward(self, x):
        """Get embedding for a single image"""
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        return F.normalize(embeddings, p=2, dim=1)  # L2 normalize
    
    def forward_pair(self, x1, x2):
        """Forward pass for pair of images"""
        emb1 = self.forward(x1)
        emb2 = self.forward(x2)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(emb1, emb2, dim=1)
        
        # Convert to probability (similarity -> [0, 1])
        similarity = (similarity + 1) / 2
        
        return similarity.unsqueeze(1)  # (batch_size, 1)


def create_dino_twin_network(
    arch: str = 'vit_small',
    patch_size: int = 8,
    pretrained_weights: Optional[str] = None,
    embedding_dimension: int = 128,
    approach: str = 'concatenation'  # 'concatenation' or 'contrastive'
) -> nn.Module:
    
    if approach == 'concatenation':
        return DINOTwinNetwork(
            arch=arch,
            patch_size=patch_size,
            pretrained_weights=pretrained_weights,
            embedding_dimension=embedding_dimension
        )
    elif approach == 'contrastive':
        return DINOContrastiveNetwork(
            arch=arch,
            patch_size=patch_size,
            pretrained_weights=pretrained_weights,
            embedding_dimension=embedding_dimension
        )
    else:
        raise ValueError(f"Unknown approach: {approach}")


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_dino_twin_network(
        arch='vit_small',
        patch_size=8,
        pretrained_weights=None,  # Set to your DINO checkpoint path
        embedding_dimension=128,
        approach='concatenation'
    ).to(device)
    
    # Test with dummy data
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 224, 224).to(device)
    x2 = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(x1, x2)
        print(f"Output shape: {output.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}") 