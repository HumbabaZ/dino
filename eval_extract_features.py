'''
 python eval_extract_features.py \
  --data_path /home/qizh093f/dino-main/train-100 \
  --pretrained_weights /home/qizh093f/dino-main/output/checkpoint0019.pth \
  --output_dir /home/qizh093f/dino-main/features
  '''

import argparse
import torch
import os
import utils
import vision_transformer as vits
from torchvision import datasets, transforms as pth_transforms
#from eval_knn import ReturnIndexDataset, extract_features
import torch.distributed as dist


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True):
    features_list = []
    indices_list = []
    
    for samples, index in data_loader:
        samples = samples.cuda(non_blocking=True)
        feats = model(samples).clone().cpu()
        features_list.append(feats)
        indices_list.append(index)
    
    # merge all the features
    all_features = torch.cat(features_list, dim=0)
    all_indices = torch.cat(indices_list, dim=0)
    
    # feature matrix
    features = torch.zeros(len(data_loader.dataset), all_features.shape[1])
    features[all_indices] = all_features
    
    return features


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, label = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx  # return images and their indicies in the dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--pretrained_weights', type=str, required=True)
    parser.add_argument('--arch', default='vit_small', type=str)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    args = parser.parse_args()

    # pre process
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    def is_tif_no_mask(path):
        return path.endswith('.tif') and not path.endswith('_mask.tif')
    #dataset = datasets.ImageFolder(args.data_path, transform=transform, is_valid_file=is_tif_no_mask)
    dataset = ReturnIndexDataset(args.data_path, transform=transform, is_valid_file=is_tif_no_mask)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    # load model and weights
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # extract feature
    features = extract_features(model, data_loader, use_cuda=True)
    features = torch.nn.functional.normalize(features, dim=1, p=2)
    labels = torch.tensor([s[-1] for s in dataset.samples]).long()

    # save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(features.cpu(), os.path.join(args.output_dir, "features.pth"))
    torch.save(labels.cpu(), os.path.join(args.output_dir, "labels.pth"))
    print("Done! Features shape:", features.shape)

if __name__ == "__main__":
    main()