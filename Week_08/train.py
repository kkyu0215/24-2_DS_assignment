import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from model import CustomCLIPClassifier
from utils import CustomDataset

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Freeze all CLIP model layers
# 일단은 classifier만 학습
for param in model.parameters():
    param.requires_grad = False


# Initialize custom classifier model
classifier_model = CustomCLIPClassifier(model).to(device)
#optimizer = torch.optim.Adam(classifier_model.classifier.parameters(), lr=1e-4)
optimizer = torch.optim.Adam([classifier_model.logit_scale], lr=1e-4)
#criterion = nn.CrossEntropyLoss()
def contrastive_loss(logits_per_image, logits_per_label):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size).to(logits_per_image.device)

    loss_image_to_label = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_label_to_image = nn.CrossEntropyLoss()(logits_per_label, labels)

    return (loss_image_to_label + loss_label_to_image) / 2

dataset = load_from_disk("/root/Week08/Representational-Learning/dataset/train")  
custom_dataset = CustomDataset(dataset, preprocess)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True)



# Training loop
classifier_model.train()
for epoch in range(5):  # Train for 5 epochs, adjust as needed
    total_loss = 0
    for images, labels in tqdm(dataloader):

        images, labels = images.to(device), labels.to(device)
        logits_per_image, logits_per_label = classifier_model(images, labels)

        optimizer.zero_grad()
        outputs = classifier_model(images, labels)
        loss = contrastive_loss(logits_per_image, logits_per_label)
        #loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the trained model
model_save_path = "/root/Week08/Representational-Learning/saved_model.pth"
torch.save(classifier_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")