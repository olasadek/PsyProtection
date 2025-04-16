#%%
import unittest
import pandas as pd
import numpy as np
import torch
import nibabel as nib
import torchvision.transforms as transforms
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from multimodal_model import MultiModalDiagnosticNet, ViTImageEncoder, EHRFeatureEncoder 
import os
#%%
class TestPipeline(unittest.TestCase):
    def setUp(self):
        """Load and prepare data using the dynamic file path"""
        
        # Load and preprocess data
        self.data = pd.read_csv("EHR_Data3.csv")

        # MRI path
        self.file = "sub-040_T1w.nii.gz"
        
        # Define columns to test
        self.numerical_cols = ['age', 'educ_yr', 'occup', 'income', 'laterality', 'amai',
                             'amai_score', 'years.begin', 'drug.age.onset', 'days.last.use',
                             'week.dose', 'tobc.lastyear', 'tobc.day', 'tobc.totyears']
        self.categorical_cols = ['illness','medication']
        
        # Define transformers
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
                dtype=np.int64
            ))
        ])
       
       # Define MRI transform with built-in normalization
        self.mri_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
])

    # Unit Tests
    def test_numeric_imputation(self):
        """Test numerical transformer pipeline"""
        print("\nRunning numeric imputation test...")
        numeric_data = self.data[self.numerical_cols]
        transformed = self.numeric_transformer.fit_transform(numeric_data)
        
        # Test 1: Shape check
        self.assertEqual(transformed.shape[1], 14)
        print("✓ Correct number of features (14)")
        
        # Test 2: Missing values check
        self.assertFalse(np.isnan(transformed).any())
        print("✓ No missing values remaining")
        
        # Test 3: Scaling check
        self.assertAlmostEqual(transformed[:, 0].mean(), 0, delta=1e-6)
        self.assertAlmostEqual(transformed[:, 0].std(), 1, delta=1e-6)
        print("✓ Features properly scaled (mean≈0, std≈1)")
        
        print(" All numeric transformation tests passed!")
        return transformed

    def test_categorical_imputation(self):
        """Test categorical transformer pipeline"""
        print("\nRunning categorical imputation test...")
        categorical_data = self.data[self.categorical_cols]
        transformed = self.categorical_transformer.fit_transform(categorical_data)
        
        # Test 1: Shape check
        self.assertEqual(transformed.shape[1], len(self.categorical_cols))
        print(f"✓ Correct number of features ({len(self.categorical_cols)})")
        
        # Test 2: Missing values check
        self.assertTrue((transformed != -1).all())
        print("✓ No missing values remaining (-1 check)")
        
        # Test 3: Data type check
        self.assertTrue(np.issubdtype(transformed.dtype, np.integer))
        print("✓ Output is integer type as expected")
        
        print(" All categorical transformation tests passed!")
        return transformed

    def test_mri_transformation(self):
        """Test MRI data transformation pipeline"""
        if not self.file:
            self.skipTest("No MRI file provided")
            return
            
        print("\nRunning MRI transformation test...")
        
        try:
            # 1. Load MRI scan
            mri_scan = nib.load(self.file)
            mri_data = mri_scan.get_fdata()
            print(f"✓ Loaded MRI data with shape: {mri_data.shape}")
            
            # 2. Convert 3D MRI data to 2D slice by reshaping the whole volume
            mri_data = mri_data.reshape(-1, mri_data .shape[1])
            print(f"✓ Reshaped MRI data to 2D with shape: {mri_data.shape}")

            # 3. Normalize the MRI data
            mri_data=(mri_data - np.min(mri_data)) / (np.max(mri_data) - np.min(mri_data))
            print("✓ Normalized MRI data")
            
            # 4. Apply transform (includes normalization)
            transformed = self.mri_transform(mri_data).float()
            
            # Verify output
            self.assertGreaterEqual(transformed.min(), 0)
            self.assertLessEqual(transformed.max(), 1)
            self.assertEqual(transformed.shape, (1, 224, 224))
            print(f"✓ transformed to shape: {transformed.shape}")
            
            print("All MRI transformation tests passed!")
            return transformed
            
        except Exception as e:
            self.fail(f"MRI transformation failed: {str(e)}")
    # Integration Tests   
    def test_vit_output_shape(self):
        """Test output shape of ViTImageEncoder using transformed MRI"""
        print("\nRunning ViT output shape test...")

        try:
           image = self.test_mri_transformation()
           if image is None:
            self.skipTest("MRI transformation did not return an image.")

           model = ViTImageEncoder()
           output = model(image.unsqueeze(0))  # Add batch dimension
           self.assertEqual(output.shape, (1, 768))
           print("✓ ViTImageEncoder output shape correct for batch size 1")

        except Exception as e:
           self.fail(f"ViT output shape test failed: {str(e)}")
    
    def test_ehr_output_shape(self):
       """Test output shape of EHRFeatureEncoder using dummy EHR input"""
       print("\nRunning EHRFeatureEncoder output shape test...")

       try:
           numeric_data = self.test_numeric_imputation()
           categorical_data = self.test_categorical_imputation()
           # Combine along feature dimension (dim=1)
           ehr_data = torch.cat([numeric_data, categorical_data], dim=1)

           model = EHRFeatureEncoder(input_dim=16)
           output = model(ehr_data)  # Pass EHR data to the model
           # Check output shape

           self.assertEqual(output.shape, (1, 50))
           print("✓ EHRFeatureEncoder output shape correct for batch size 1")

       except Exception as e:
          self.fail(f"EHR output shape test failed: {str(e)}")

    def test_multimodal_diagnostic_net(self):
       """Test MultiModalDiagnosticNet end-to-end with dummy MRI and EHR input."""
       try:
           print("\nRunning MultiModalDiagnosticNet integration test...")

           # Prepare MRI image
           mri_image = self.test_mri_transformation()  # shape: (1, 224, 224)
           mri_image = mri_image.unsqueeze(0)  # Add batch dimension -> (1, 1, 224, 224)

           # Prepare EHR data
           numeric_data = self.test_numeric_imputation()  # (N, num_features)
           categorical_data = self.test_categorical_imputation()  # (N, cat_features)
        
           ehr_data_np = np.concatenate([numeric_data, categorical_data], axis=1)
           ehr_data = torch.tensor(ehr_data_np, dtype=torch.float32)
           ehr_data = ehr_data[:1]  # Ensure batch size is 1

           # Model init
           ehr_input_dim = ehr_data.shape[1]
           model = MultiModalDiagnosticNet(ehr_input_dim=ehr_input_dim)

           # Forward pass
           output = model(mri_image, ehr_data)

           # Assertions
           self.assertEqual(output.shape, (1, 1))
           self.assertGreaterEqual(output.item(), 0.0)
           self.assertLessEqual(output.item(), 1.0)
           print("✓ MultiModalDiagnosticNet output shape correct and within [0, 1]")
   
           print("All MultiModalDiagnosticNet integration tests passed!")

       except Exception as e:
           self.fail(f"MultiModalDiagnosticNet test failed: {str(e)}")
           
if __name__ == '__main__':
    print("Starting test suite...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print("\nAll tests completed!")
# %%
