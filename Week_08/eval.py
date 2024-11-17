import torch
import clip
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_from_disk

from model import CustomCLIPClassifier
from utils import CustomDataset, compute_ece, plot_confidence_and_accuracy, visualize_embeddings_with_tsne

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the saved model
model_load_path = "/root/Week08/Representational-Learning/saved_model.pth"
classifier_model = CustomCLIPClassifier(model).to(device)
classifier_model.load_state_dict(torch.load(model_load_path))
classifier_model.eval()
print(f"Model loaded from {model_load_path}")

# Load and preprocess data
dataset = load_from_disk("/root/Week08/Representational-Learning/dataset/val")
custom_dataset = CustomDataset(dataset, preprocess)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=False)

# Evaluate model and measure ECE
all_probs = []
all_labels = []
with torch.no_grad():
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits_per_image, logits_per_label = classifier_model(images, labels)
        #outputs = classifier_model(images, labels)
        probs = torch.softmax(logits_per_image, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# breakpoint()
# print(f"Batch sizes (probs): {[batch.shape for batch in all_probs]}")  # 각 배치 크기 확인
# print(f"Batch sizes (labels): {[batch.shape for batch in all_labels]}")  # 각 배치 크기 확인

all_probs = np.concatenate(all_probs[:-1])
all_labels = np.concatenate(all_labels[:-1])

ece_score = compute_ece(all_probs, all_labels)
print(f"ECE Score: {ece_score:.4f}")
visualize_embeddings_with_tsne(classifier_model, dataloader)
plot_confidence_and_accuracy(all_probs, all_labels)