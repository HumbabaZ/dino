import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# load
features_path = "/home/qizh093f/dino-main/features/features.pth"
labels_path = "/home/qizh093f/dino-main/features/labels.pth"
features = torch.load(features_path)
labels = torch.load(labels_path).numpy()

# normalization
features = features / (features.norm(dim=1, keepdim=True) + 1e-10)
features = features.cpu().numpy()

# sort
sorted_indices = np.argsort(labels)
features_sorted = features[sorted_indices]
labels_sorted = labels[sorted_indices]

# cos similarity
sim_matrix = np.dot(features_sorted, features_sorted.T)

same = []
diff = []
for i in range(len(labels_sorted)):
    for j in range(len(labels_sorted)):
        if i == j:
            continue
        if labels_sorted[i] == labels_sorted[j]:
            same.append(sim_matrix[i, j])
        else:
            diff.append(sim_matrix[i, j])

#print("同类平均相似度:", np.mean(same))
#print("异类平均相似度:", np.mean(diff))
same_avg = np.mean(same)
diff_avg = np.mean(diff)


# 5. visualize
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, cmap='viridis')
plt.title('Cosine Similarity Matrix (sorted by class)')
plt.xlabel('Image Index')
plt.ylabel('Image Index')

plt.text(10, -5, f"Intra-class Average Similarity: {same_avg:.4f}", 
         fontsize=12, ha='left')
plt.text(60, -5, f"Inter-class Average Similarity: {diff_avg:.4f}", 
         fontsize=12, ha='left')

plt.tight_layout()

os.makedirs("output", exist_ok=True)
plt.savefig("output/similarity_matrix.png", dpi=300)

# plt.show()