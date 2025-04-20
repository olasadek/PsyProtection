import { useState } from 'react';

const MRIUpload = ({ onUpload, patientId }) => {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = () => {
    if (!file) return;
    onUpload(file);
  };

  return (
    <div className="mri-upload">
      <h2>MRI Upload</h2>
      <div className="upload-area">
        <input 
          type="file" 
          id="mri-upload" 
          accept=".nii,.nii.gz" 
          onChange={handleFileChange}
          disabled={isUploading}
        />
        <label htmlFor="mri-upload">
          {file ? file.name : 'Choose MRI file (NIfTI format)'}
        </label>
        
        {file && (
          <button 
            onClick={handleUpload} 
            disabled={isUploading}
            className="upload-btn"
          >
            Upload MRI
          </button>
        )}
      </div>
    </div>
  );
};

export default MRIUpload;