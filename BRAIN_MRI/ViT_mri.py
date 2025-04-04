import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification
import pandas as pd
import os 

class MRIDataset(Dataset):
    def __init__(self, labels_df, root_dir, transform=None):
        self.labels = labels_df
        self.root_dir = root_dir
        self.transform = transform
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.labels.iloc[idx]['filename'])
        img = nib.load(img_path).get_fdata()
        
        
        img_slice = img[:, :, img.shape[2]//2]  
        
        
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        img_rgb = np.stack([img_slice]*3, axis=0)  
        
        
        inputs = self.processor(images=img_rgb, return_tensors="pt")
        label = 1 if self.labels.iloc[idx]['label'] == 'drug_abuser' else 0
        
        return inputs['pixel_values'].squeeze(), torch.tensor(label)


labels = pd.read_csv("brain_mri/labels.csv")
dataset = MRIDataset(labels, root_dir="brain_mri")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2,
    id2label={0: "non_abuser", 1: "drug_abuser"},
    ignore_mismatched_sizes=True  
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(50):
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).logits
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
