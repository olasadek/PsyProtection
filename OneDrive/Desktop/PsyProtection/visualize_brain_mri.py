# ======================
# MRI ANALYSIS PIPELINE
# ======================

import os
import zipfile
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


zip_path = "mri_dataset.zip"
extract_dir = "brain_mri"
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Dataset extracted!")
else:
    print("Dataset already exists.")


labels = pd.read_csv(f"{extract_dir}/labels.csv")
print("\nFirst 5 labels:")
print(labels.head())


def plot_sample(filepath, title):
    """Plot axial, sagittal, and coronal slices"""
    img = nib.load(filepath).get_fdata()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial (Top-Bottom)
    axes[0].imshow(img[:, :, img.shape[2]//2], cmap='gray')
    axes[0].set_title(f"Axial: {title}")
    
    # Sagittal (Side)
    axes[1].imshow(img[:, img.shape[1]//2, :], cmap='gray')
    axes[1].set_title(f"Sagittal: {title}")
    
    # Coronal (Front)
    axes[2].imshow(img[img.shape[0]//2, :, :], cmap='gray')
    axes[2].set_title(f"Coronal: {title}")
    
    plt.show()

print("\nVisualizing samples...")
plot_sample(f"{extract_dir}/{labels[labels['label']=='drug_abuser'].iloc[0]['filename']}", "Drug Abuser")
plot_sample(f"{extract_dir}/{labels[labels['label']=='non_abuser'].iloc[0]['filename']}", "Control")


def extract_features(img_data):
    """Calculate simulated biomarkers"""
    return {
        'frontal_cortex': np.mean(img_data[20:30, 10:50, 25:35]),  # Lower in abusers
        'ventricle_size': np.sum(img_data[10:20, 30:40, 20:30] < -0.5),  # Larger in abusers
        'hippocampus': np.mean(img_data[15:25, 30:40, 20:30])  # Smaller in abusers
    }


features = []
for _, row in labels.iterrows():
    img = nib.load(f"{extract_dir}/{row['filename']}").get_fdata()
    feat = extract_features(img)
    feat['label'] = row['label']
    features.append(feat)

feature_df = pd.DataFrame(features)
print("\nExtracted features:")
print(feature_df.groupby('label').mean())


X = feature_df[['frontal_cortex', 'ventricle_size', 'hippocampus']]
y = feature_df['label'].map({'drug_abuser': 1, 'non_abuser': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(f"\nModel accuracy: {clf.score(X_test, y_test):.2f}")

# 6. Save results
feature_df.to_csv("mri_features.csv", index=False)
print("\nAnalysis complete! Saved features to mri_features.csv")