"""
Training script for DINO-Twin Network
Modified from the original twin network training script to use DINO backbone
"""

from __future__ import print_function

import argparse
import os
import shutil
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

# Import the DINO-Twin Network
from dino_twin_network import create_dino_twin_network, DINOTwinNetwork

# Import original components (assuming they exist in the same directory or can be imported)
# If these modules don't exist, you'll need to create them or modify the imports
try:
    from callbacks import CallbackContainer, LRScheduleCallback, TimeCallback, CSVLogger, ModelCheckpoint, MonitorMode
    from datasets import RandomExclusiveListApply, RandomApply, RandomOrganoidPairDataset, DeterministicOrganoidPairDataset, \
        SquarePad, OnlineRandomOrganoidHistPairDataset, OnlineDeterministicOrganoidHistPairDataset
    from metrics import BinaryAccuracy, MetricContainer, ConfusionMatrix
except ImportError:
    print("Warning: Some callback/dataset/metric modules not found. Using basic implementations.")
    # Basic implementations as fallback
    class CallbackContainer:
        def __init__(self, callbacks=None):
            self.callbacks = callbacks or []
        def on_epoch_begin(self, epoch, logs): pass
        def on_epoch_end(self, epoch, logs): return logs
    
    class MetricContainer:
        def __init__(self, metrics=None):
            self.metrics = metrics or []
        def set_train(self): pass
        def set_val(self): pass
        def update(self, outputs, targets): pass
        def summary_string(self): return ""
        def summary(self): return {}
        def reset(self): pass


class DINOTwinTrainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 callbacks: CallbackContainer = None,
                 metrics: MetricContainer = None,
                 log_interval=10,
                 dry_run=False
                 ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks or CallbackContainer()
        self.metrics = metrics or MetricContainer()
        self.log_interval = log_interval
        self.dry_run = dry_run

    def train(self,
              epochs: int,
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader = None
              ):

        for epoch in range(1, epochs + 1):
            logs = {}
            self.callbacks.on_epoch_begin(epoch, logs)
            train_logs = self.train_epoch(epoch, train_loader)
            val_logs = self.val_epoch(epoch, val_loader) if val_loader else {}
            logs.update(train_logs)
            logs.update(val_logs)
            logs = self.callbacks.on_epoch_end(epoch, logs)
            if "stop_training" in logs.keys() and logs["stop_training"]:
                break

    def train_epoch(self, epoch, train_loader: torch.utils.data.DataLoader):
        self.model.train()
        self.metrics.set_train()

        logs = {
            "epoch": epoch,
            "loss": 0,
        }

        batch_idx = 0
        for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
            images_1, images_2, targets = images_1.to(self.device), images_2.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass through DINO-Twin Network
            outputs = self.model(images_1, images_2)

            self.metrics.update(outputs, targets)
            loss = self.criterion(outputs, targets.float())  # Ensure targets are float for BCELoss
            logs["loss"] += loss.item()

            loss.backward()
            self.optimizer.step()
            
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t {}'.format(
                    epoch, batch_idx * len(images_1), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           logs["loss"] / (batch_idx + 1), self.metrics.summary_string()
                ))
                if self.dry_run:
                    break
        logs["loss"] /= (batch_idx + 1)

        logs.update(self.metrics.summary())
        self.metrics.reset()
        return logs

    def val_epoch(self, epoch, val_loader: torch.utils.data.DataLoader) -> Dict:
        self.model.eval()
        self.metrics.set_val()

        logs = {
            "val_loss": 0,
            "epoch": epoch
        }

        with torch.no_grad():
            for batch_idx, (images_1, images_2, targets) in enumerate(val_loader):
                images_1, images_2, targets = images_1.to(self.device), images_2.to(self.device), targets.to(
                    self.device)
                outputs = self.model(images_1, images_2)

                logs["val_loss"] += self.criterion(outputs, targets.float()).item()

                self.metrics.update(outputs, targets)
                
                if batch_idx % self.log_interval == 0:
                    print('Val Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                        epoch, batch_idx * len(images_1), len(val_loader.dataset),
                               100. * batch_idx / len(val_loader)))
                if self.dry_run:
                    break

        logs["val_loss"] /= (batch_idx + 1)

        print('\nTest set: Average Loss: {:.4f}\t {}'.format(logs["val_loss"], self.metrics.summary_string() + "\n"))

        logs.update(self.metrics.summary())
        self.metrics.reset()

        return logs


def get_basic_metrics() -> MetricContainer:
    """Basic metrics implementation if the original metrics module is not available"""
    try:
        return MetricContainer([
            BinaryAccuracy(name='accuracy', precision=3),
            ConfusionMatrix(pos_min=0.5, pos_max=1.0, precision=0)
        ])
    except:
        return MetricContainer()


def create_simple_dataset(data_dir, transforms=None, batch_size=64, steps_per_epoch=100):
    """
    Simple dataset creation if the original datasets are not available
    You'll need to adapt this to your specific dataset structure
    """
    try:
        # Try to use original datasets
        from datasets import RandomOrganoidPairDataset, DeterministicOrganoidPairDataset, SquarePad
        
        train_dataset = RandomOrganoidPairDataset(
            data_dir,
            num_batches=steps_per_epoch,
            batch_size=batch_size,
            transforms=transforms
        )
        return train_dataset
    except ImportError:
        # Fallback: create a simple ImageFolder-based dataset
        print("Original datasets not found. Please implement your dataset loading logic here.")
        raise NotImplementedError("Please implement dataset loading for your specific use case")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DINO-Twin Network for Organoids')
    parser.add_argument('--data-dir',
                        type=str,
                        default="../data/train-100",
                        help="Data directories for input files"
                        )
    parser.add_argument('--val-data-dir',
                        type=str,
                        default=None,
                        help="Optional validation data directory"
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,  # Reduced for ViT
                        metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--val-batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-name',
                        type=str,
                        help='model name',
                        default="dino_twin_test"
                        )
    parser.add_argument('--override',
                        help='Whether to override models under the same name',
                        action='store_true',
                        default=False
                        )
    parser.add_argument('--embedding-dimension',
                        type=int,
                        default=128
                        )
    parser.add_argument('--model-dir',
                        type=str,
                        default="./dino-twin-models",
                        help="Where to save checkpoints and such."
                        )
    parser.add_argument('--total-steps',
                        type=int,
                        default=6000,
                        help='How many steps to train for in total'
                        )
    parser.add_argument('--steps-per-epoch',
                        type=int,
                        default=100,
                        help='How many batches to train for per epoch'
                        )
    
    # DINO-specific arguments
    parser.add_argument('--dino-arch',
                        type=str,
                        default='vit_small',
                        choices=['vit_tiny', 'vit_small', 'vit_base'],
                        help='DINO ViT architecture'
                        )
    parser.add_argument('--dino-patch-size',
                        type=int,
                        default=8,
                        help='Patch size for DINO ViT'
                        )
    parser.add_argument('--dino-weights',
                        type=str,
                        default=None,
                        help='Path to pre-trained DINO weights'
                        )
    parser.add_argument('--freeze-backbone',
                        action='store_true',
                        default=False,
                        help='Freeze DINO backbone during training'
                        )
    parser.add_argument('--approach',
                        type=str,
                        default='concatenation',
                        choices=['concatenation', 'contrastive'],
                        help='Twin network approach'
                        )
    
    args = parser.parse_args()

    if args.val_data_dir is None:
        args.val_data_dir = args.data_dir

    epochs = args.total_steps // args.steps_per_epoch

    model_dir = os.path.join(args.model_dir, args.model_name)
    if args.override and os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "dino_twin_network.pt")

    if os.path.exists(model_path) and not args.override:
        print("Model exists. Specify --override to override")
        exit()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if use_cuda:
        print("Using CUDA device...")
        device = torch.device("cuda")
    else:
        print("Using CPU...")
        device = torch.device("cpu")

    # Data loading (you'll need to implement this based on your dataset structure)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),  # ViT typically uses 224x224
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # You'll need to implement dataset loading here based on your specific dataset
    # This is a placeholder - adapt to your dataset structure
    print("Note: You need to implement dataset loading based on your specific dataset structure")
    
    # Create DINO-Twin Network model
    model = create_dino_twin_network(
        arch=args.dino_arch,
        patch_size=args.dino_patch_size,
        pretrained_weights=args.dino_weights,
        embedding_dimension=args.embedding_dimension,
        approach=args.approach
    ).to(device)

    # Set backbone to be frozen if specified
    if args.freeze_backbone and hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("DINO backbone frozen")

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        weight_decay=0.01
    )

    # Metrics
    metrics = get_basic_metrics()

    # Create dummy data loaders for testing (replace with your actual dataset)
    print("Creating dummy data for testing. Replace with your actual dataset loading.")
    
    # Dummy dataset for testing
    dummy_train_data = torch.utils.data.TensorDataset(
        torch.randn(1000, 3, 224, 224),  # images_1
        torch.randn(1000, 3, 224, 224),  # images_2
        torch.randint(0, 2, (1000,))     # targets
    )
    dummy_val_data = torch.utils.data.TensorDataset(
        torch.randn(200, 3, 224, 224),
        torch.randn(200, 3, 224, 224),
        torch.randint(0, 2, (200,))
    )
    
    train_loader = torch.utils.data.DataLoader(
        dummy_train_data, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dummy_val_data,
        batch_size=args.val_batch_size,
        shuffle=False
    )

    # Trainer
    trainer = DINOTwinTrainer(
        model=model,
        log_interval=10,
        optimizer=optimizer,
        criterion=nn.BCELoss(),
        device=device,
        metrics=metrics,
        callbacks=CallbackContainer()  # Add actual callbacks if available
    )

    print(f"Starting training for {epochs} epochs...")
    trainer.train(epochs, train_loader, val_loader)

    # Save model
    if hasattr(model, 'kwargs'):
        torch.save([model.kwargs, model.state_dict()], model_path)
    else:
        torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main() 