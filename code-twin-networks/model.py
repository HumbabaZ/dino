import torch
import torch.nn as nn
import torchvision
from enum import Enum
import os
import sys

# 添加DINO相关导入
# 假设vision_transformer.py在父目录或者当前目录
try:
    import vision_transformer as vits
except ImportError:
    # 如果vision_transformer不在当前路径，尝试从父目录导入
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import vision_transformer as vits

from resnet18_1d import resnet18_1d


class InputType(Enum):
    IMAGES = "images"
    DINO = "dino"  # 添加DINO类型，移除HISTOGRAM


class Concatenation(torch.nn.Module):
    def __init__(self):
        super(Concatenation, self).__init__()

    def forward(self, x1, x2):
        return torch.cat([x1, x2], 1)


def load_dino_model(checkpoint_path, arch='vit_small', patch_size=16):
    """加载预训练的DINO模型"""
    print(f"Loading DINO model: {arch} with patch size {patch_size}")
    print(f"Checkpoint path: {checkpoint_path}")
    
    # 创建模型
    if arch not in vits.__dict__:
        raise ValueError(f"Architecture {arch} not found in vision_transformer module")
    
    model = vits.__dict__[arch](patch_size=patch_size)
    
    # 加载checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"DINO checkpoint not found at: {checkpoint_path}")
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 提取student的state_dict
    if 'student' in checkpoint:
        print("Found 'student' in checkpoint")
        state_dict = checkpoint['student']
        # 移除DDP wrapper的前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # 只保留backbone部分，移除head部分
        backbone_state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('head')}
        model.load_state_dict(backbone_state_dict, strict=False)
        print(f"Loaded {len(backbone_state_dict)} parameters from student")
    else:
        print("Loading full checkpoint")
        # 如果没有student键，尝试直接加载
        state_dict = {k: v for k, v in checkpoint.items() 
                     if not k.startswith('head')}
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {len(state_dict)} parameters")
    
    print("DINO model loaded successfully")
    return model


def get_dino_embed_dim(arch):
    """获取DINO架构的嵌入维度"""
    if arch == 'vit_tiny':
        return 192
    elif arch == 'vit_small':
        return 384
    elif arch == 'vit_base':
        return 768
    else:
        raise ValueError(f"Unsupported DINO architecture: {arch}")


def build_embedding_network(
        input_type: InputType,
        embedding_dimension: int,
        dino_checkpoint_path: str = None,
        dino_arch: str = 'vit_small',
        dino_patch_size: int = 16
):
    if input_type == InputType.IMAGES:
        print("Building ResNet18 embedding network")
        embedding_network = torchvision.models.resnet18()
        fc_in_features = embedding_network.fc.in_features
        embedding_network = nn.Sequential(*(list(embedding_network.children())[:-1]))
        embedding_network = nn.Sequential(
            embedding_network,
            torch.nn.Flatten(),
            torch.nn.Linear(fc_in_features, embedding_dimension)
        )
        
    elif input_type == InputType.DINO:
        print("Building DINO embedding network")
        # 加载预训练的DINO模型
        if dino_checkpoint_path is None:
            raise ValueError("DINO checkpoint path must be provided for DINO input type")
        
        dino_model = load_dino_model(dino_checkpoint_path, dino_arch, dino_patch_size)
        
        # 获取DINO的输出维度
        dino_embed_dim = get_dino_embed_dim(dino_arch)
        print(f"DINO embedding dimension: {dino_embed_dim}")
        print(f"Target embedding dimension: {embedding_dimension}")
        
        # 冻结DINO参数（可选：如果想要fine-tune，可以注释掉这部分）
        for param in dino_model.parameters():
            param.requires_grad = False
        print("DINO parameters frozen")
        
        # 构建嵌入网络
        if embedding_dimension == dino_embed_dim:
            # 如果目标维度与DINO输出维度相同，直接使用
            print("Using DINO output directly (no projection layer)")
            embedding_network = dino_model
        else:
            # 如果不同，添加投影层
            print(f"Adding projection layer: {dino_embed_dim} -> {embedding_dimension}")
            embedding_network = nn.Sequential(
                dino_model,
                nn.Linear(dino_embed_dim, embedding_dimension)
            )
    else:
        raise NotImplementedError(f"Embedding network for input type {input_type} not implemented")

    return embedding_network


class SiameseNetwork(nn.Module):
    def __init__(self,
                 input_type: InputType,
                 embedding_dimension: int,
                 dino_checkpoint_path: str = None,
                 dino_arch: str = 'vit_small',
                 dino_patch_size: int = 16
                 ):
        super(SiameseNetwork, self).__init__()

        self.kwargs = {
            'input_type': input_type,
            'embedding_dimension': embedding_dimension,
            'dino_checkpoint_path': dino_checkpoint_path,
            'dino_arch': dino_arch,
            'dino_patch_size': dino_patch_size
        }

        print(f"Initializing SiameseNetwork with:")
        print(f"  - Input type: {input_type}")
        print(f"  - Embedding dimension: {embedding_dimension}")
        if input_type == InputType.DINO:
            print(f"  - DINO architecture: {dino_arch}")
            print(f"  - DINO patch size: {dino_patch_size}")
            print(f"  - DINO checkpoint: {dino_checkpoint_path}")

        self.embedding = build_embedding_network(
            input_type,
            embedding_dimension,
            dino_checkpoint_path,
            dino_arch,
            dino_patch_size
        )

        # 初始化权重（不包括预训练的DINO部分）
        if input_type != InputType.DINO:
            print("Initializing embedding network weights")
            self.embedding.apply(self.init_weights)
        else:
            # 只初始化投影层（如果存在）
            dino_embed_dim = get_dino_embed_dim(dino_arch)
            if embedding_dimension != dino_embed_dim:
                print("Initializing projection layer weights")
                self.embedding[-1].apply(self.init_weights)

        # 构建分类头
        print("Building classification head")
        self.merge_layer = Concatenation()
        self.fc = nn.Sequential(
            nn.Linear(2 * embedding_dimension, embedding_dimension),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dimension, 1),
            nn.Flatten()
        )
        self.sigmoid = nn.Sigmoid()

        print("Initializing classification head weights")
        self.fc.apply(self.init_weights)
        
        print("SiameseNetwork initialization complete")

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.embedding(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # 获取两个输入的嵌入
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        if self.merge_layer:
            outputs = self.merge_layer(output1, output2)
            outputs = self.fc.forward(outputs)
            outputs = self.sigmoid(outputs)
            return outputs.view(outputs.size()[0])

        return output1, output2

    @property
    def embedding_network(self):
        return self.embedding

    @property
    def args(self):
        return self.kwargs