# Load the dataset from S3 (Cloud Storage)
# Recommended: Use Google Colab for handling large datasets
# Install AWS CLI if not already installed:
# pip install awscli
# Sync the dataset from OpenNeuro S3 bucket (public access, no credentials needed):
# aws s3 sync --no-sign-request s3://openneuro.org/ds003346 ds003346-download/
#%%
# Import necessary libraries
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.impute import SimpleImputer
import shutil
from sklearn.preprocessing import  StandardScaler,OrdinalEncoder
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import joblib
from multimodal_model import MultiModalDiagnosticNet, ViTImageEncoder, EHRFeatureEncoder
#%%
# Plot a slice of an MRI image for a patient to ensure that the data is downloaded correctly.

# Load MRI scan
mri_path = "C:\AI in Industry\Final Project\MRI\sub-040_T1w.nii.gz"
mri_scan = nib.load(mri_path)
mri_data = mri_scan.get_fdata()  # Convert to NumPy array
print(mri_data.shape)


plt.imshow(mri_data[:, :,50], cmap="gray")
plt.title("MRI Slice (Axial View)")
plt.axis("off")
plt.show()
#%%

# Preprocessing the Candidates EHR Data

# Load EHR data
data = pd.read_csv("EHR_Data2.txt")

data["illness"]=data["illness"].fillna('non')
data["medication"]=data["medication"].fillna('non')

# Specify the patient data that is not classified and delete it
nan_indices = np.where(data['group'].isna())
data_cleaned = data.dropna(subset=['group'])
data_cleaned

# Drop the unnecessary features
columns_to_drop = ['participant_id','sex','civil_st', 'child', 'admin.route','child_num', 'work_thirtydays', 'work_threeyears','ed_score','comm','educ',"exclusion"]
New_data = data_cleaned.drop(columns=columns_to_drop)
New_data.head()

"""# Overview of the Dataset"""

New_data['income'] = pd.to_numeric(New_data['income'], errors='coerce')

# Dataset info
print("\nDataset Information:")
print(New_data.info())

# Check for missing values
print("\nMissing Values in Each Column:")
print(New_data.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(New_data.describe())

"""# Class Distribution"""

# Check class distribution
print("\nClass Distribution:")
print(New_data["group"].value_counts())

# Visualize class distribution
sns.countplot(data=data, x="group", palette="viridis")
plt.title("Class Distribution")
plt.show()

""" Data Cleaning"""

# Encode target variable
New_data["group"] = New_data["group"].map({1: 0, 2: 1})

""" Handle Missing Values"""

# Columns with missing values
missing_cols = [ 'laterality', 'age', 'educ_yr', 'occup', 'income', 'amai',
       'amai_score', 'years.begin', 'drug.age.onset', 'days.last.use',
       'week.dose', 'tobc.lastyear', 'tobc.day', 'tobc.totyears']

# Impute missing values with the most frequent value
imputer = SimpleImputer(strategy="most_frequent")
New_data[missing_cols] = imputer.fit_transform(New_data[missing_cols])

# Verify missing values are handled
print("\nMissing Values After Imputation:")
print(New_data.isnull().sum())

""" Split Features and Target"""

# Split features and target variable
X = New_data.drop("group", axis=1)
y = New_data["group"]

# Display dataset shapes
print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

""" Delete the ambiguous participant from the directory"""

Mis_data=data.iloc[nan_indices]["participant_id"]
directory= "/content/ds003346-download"
for folder in Mis_data:
  folder_path = os.path.join(directory,folder)
  shutil.rmtree(folder_path)

""" Spliting and Preprocessing the data"""

# Spliting the data(both the mri image, EHR data and the labels)
def split_dataset(directory, X, y,  split=0.9):
    # List folders
    folders = [f for f in sorted(os.listdir(directory))[11:]
               if os.path.isdir(os.path.join(directory, f))]

    # Split index
    split_index = int(len(folders) * split)

    # Train-test split
    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y[:split_index]
    y_test = y[split_index:]

    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Scale numeric features
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Encode categorical features as integers (0, 1, 2, ...)
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

    # Split folders and slices
    train_folders = folders[:split_index]
    test_folders = folders[split_index:]

    return train_folders, test_folders,X_train,X_test,y_train,y_test

train_folders, test_folders,X_train,X_test,y_train,y_test=split_dataset(directory,X,y, split=0.8)

# Preproces the MRI data to match the model we are going to use.
# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])
def Extract_T1_RMI_dataset(directory,Specific_file):
# Dictionary to store MRI data for each subject
 mri_data_dict = {}

# Loop through each subject folder
 for sub in Specific_file:
    sub_path = os.path.join(directory, sub, "anat")  # Path to anat folder
    if os.path.exists(sub_path):
        # Find the T1-weighted MRI file
        for file in os.listdir(sub_path):
            if file.endswith("_T1w.nii.gz"):  # Ensure it's a T1-weighted scan
                mri_path = os.path.join(sub_path, file)


                # Load the MRI scan
                mri_scan = nib.load(mri_path)
                mri_data = mri_scan.get_fdata()  # Extract MRI image data
                # Normalize
                mri_data = mri_data.reshape(-1, mri_data .shape[1]) # Get middle slice
                print(mri_data.shape)
                mri_data=(mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
                # Extract some slices from the existing ones to avoid redundant.

                #mri_data = np.stack([mri_data] * 3, axis=-1) # Convert to 3-channel (Grayscale → RGB
                # Transform image to a suitable fom.

                mri_data=transform(mri_data)
                # Store in dictionary
                mri_data_dict[sub] = mri_data


                print(f"Loaded MRI scan for {sub}: {mri_data.shape}")

  # Check if data was loaded
 print(f"Total MRI scans loaded: {len(mri_data_dict)}")
 return  mri_data_dict

mri_data_dict_train=Extract_T1_RMI_dataset(directory,train_folders)

mri_data_dict_test=Extract_T1_RMI_dataset(directory,test_folders)

""" Create data loaders for training and testing datasets."""

images_train = np.array(list(mri_data_dict_train.values()))
images_train=torch.from_numpy(images_train).float()
images_test = np.array(list(mri_data_dict_test.values()))
images_test=torch.from_numpy(images_test).float()
X_train1=torch.from_numpy(X_train.to_numpy()).float()
X_test1=torch.from_numpy(X_test.to_numpy()).float()
y_train1=torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
y_test1=torch.tensor(y_test.to_numpy(), dtype=torch.float).unsqueeze(1)

# Create a Train TensorDataset
train_dataset = TensorDataset(images_train,X_train1,y_train1)
train_dataloader  = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

# Create a Test TensorDataset
test_dataset = TensorDataset(images_test,X_test1,y_test1)
test_dataloader  = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

"""To train a robust multi-model for this task, we implemented a combined architecture consisting of two distinct networks:


1.   **Vits-based feature extractor**: This model processes MRI images and extracts 768 high-level features that encapsulate relevant spatial and structural information.
2.   **Neural network for EHR data** : A separate neural network is designed to extract 50 informative features from Electronic Health Records (EHR), ensuring that patient metadata contributes meaningfully to the final prediction

Both feature sets are **concatenated and used as input** to the final predictive layer. The training process was structured as follows:



*  The combined model was trained using the **train and test data loaders**, ensuring efficient mini-batch processing
*   **Binary Cross Entropy Loss** (BCELoss) was used as the criterion for optimization.

*  The **Adam optimizer** with weight decay was applied to fine-tune model parameters.
*   The **best validation loss** was tracked, and the optimal model state was saved.

*   The finalized model, incorporating stacked features from both sources, was stored  for future predictions.

# Train and Test the Model
"""


# ====================== 1. Training & Validation ======================

def calculate_accuracy(outputs, labels):
    preds = (outputs > 0.5).float()
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def train_step(model, criterion, optimizer, images, features, labels, device):
    model.train()
    images, features, labels = images.to(device), features.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images, features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    acc = calculate_accuracy(outputs, labels)
    return loss.item(), acc

def validation_step(model, criterion, images, features, labels, device):
    model.eval()
    with torch.no_grad():
        images, features, labels = images.to(device), features.to(device), labels.to(device)
        outputs = model(images, features)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)
    return loss.item(), acc

def train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, epochs, device):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_loss = float('inf')
    best_model_state = None

    model.to(device)

    for epoch in range(epochs):
        epoch_train_loss, epoch_train_acc = 0.0, 0.0
        for images, features, labels in train_loader:
            loss, acc = train_step(model, criterion, optimizer, images, features, labels, device)
            epoch_train_loss += loss
            epoch_train_acc += acc

        train_losses.append(epoch_train_loss / len(train_loader))
        train_accs.append(epoch_train_acc / len(train_loader))

        epoch_val_loss, epoch_val_acc = 0.0, 0.0
        for images, features, labels in test_loader:
            loss, acc = validation_step(model, criterion, images, features, labels, device)
            epoch_val_loss += loss
            epoch_val_acc += acc

        val_losses.append(epoch_val_loss / len(test_loader))
        val_accs.append(epoch_val_acc / len(test_loader))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_state = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch+1:02d} | '
              f'Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.4f} | '
              f'Val Loss: {val_losses[-1]:.4f} | Val Acc: {val_accs[-1]:.4f}')

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_state': best_model_state,
        'best_val_loss': best_val_loss
    }


# ====================== 2. Prediction ======================

def predict_proba(model, images, features, device):
    model.eval()
    with torch.no_grad():
        images, features = images.to(device), features.to(device)
        outputs = model(images, features)
    return outputs.squeeze().cpu().numpy()



# Load your data: Replace these tensors with your real data
# images_train, X_train1, y_train
# images_test,  X_test1,  y_test

# Example placeholders:
# images_train = torch.randn(100, 1, 244, 244)
# ehr_train = torch.randn(100, 20)
# y_train = torch.randint(0, 2, (100, 1)).float()
# images_test = torch.randn(20, 1, 244, 244)
# ehr_test = torch.randn(20, 20)
# y_test = torch.randint(0, 2, (20, 1)).float()

train_dataset = TensorDataset(images_train, X_train1, y_train1)
test_dataset = TensorDataset(images_test, X_test1, y_test1)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

feature_dim = X_train1.shape[1]
model = MultiModalDiagnosticNet(feature_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, epochs, device)

# Save best model
best_model = MultiModalDiagnosticNet(feature_dim)
best_model.load_state_dict(results['best_state'])
best_model.to('cpu')  # Move to CPU before saving
joblib.dump(best_model, "multi_modal_diagnostic_model.pkl")

print("Model training complete and saved.")

# Load the trained model (this assumes the model file is in the same directory as the script)
model = joblib.load("multi_modal_diagnostic_model.pkl")

# Get 2 samples from your test data (or create new ones)
sample1_image = images_test[2]    # First test image
sample1_features = X_test1[2]     # First test features

sample2_image = images_test[3]    # Second test image
sample2_features = X_test1[3]

# Stack images and features together
batch_images = torch.stack([sample1_image, sample2_image])      # Shape: [2, 1, H, W]
batch_features = torch.stack([sample1_features, sample2_features])

with torch.no_grad():
    # Get model predictions
    predictions =model(batch_images, batch_features)  # Pass both images and features to the model

    # Get predicted classes: threshold at 0.5 to decide class (0 or 1)
    predicted_classes = (predictions >= 0.5).float()  # 1 if probability >= 0.5 else 0

    print("Predicted Probabilities:", predictions)
    print("Predicted Classes:", predicted_classes)




# %%
