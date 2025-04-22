# Psychiatric Care

A web portal to help assist the dependency of psychiatric patients on their perscribed medicines


## Table of Contents

- [Features](#features)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)

## Features

## Features

- **Multimodal Model**: 
  - Processes **Eelectronic health record data** (Dense Neural Network) and **3D MRI volume images** (3D ViT).
  - Combines features using a **Late Fusion Classifier**.
  - **Training Accuracy**: 97%, **Testing Accuracy**: 87%.

- **Explainable AI**:
  - Generates a **heatmap** using occlusion-based Grad-CAM analysis to highlight important brain areas related to addiction risk.
  - Provides transparency by showing which parts of the MRI contribute to the modelâ€™s prediction.

- **RAG System**:
  - Retrieves research articles from **PMC** and **ArXiv** based on user queries.
  - Articles are chunked, embedded, and stored in a **vector database**.
  - Uses a **GPT model** to rewrite queries and generate answers based on the research.
  - **Cached in TinyDB** for fast responses to repeated queries.
** Automated Pipeline is implemented if the prediction of the patient is drug abuser where the explainable AI component will give back
   a heatmap and a query will form using his medication and illness and thrown into the rag system for recommendation treatment**

## Datasets

This project utilizes the following datasets:

1. **Electronic health care and MRI Dataset**
   - **Description**: A wide collection of Brain MRI with the corresponding health data.
   - **Source**: OpenNeuro addiction dataset.
   - **Usage**: Used to train the multimodal prediction model.

2. **Articles dataset**
   - **Description**: Contains text data from articles .
   - **Source**:  from both pubmed and arxiv via retrieval tools.
   - **Usage**: Utilized for augmentation and answer generation for the rag system.

## Project Structure

- `EEP\`: contains the eep server code that orchastrates the ports.
- `rag.py\`: Contains the logic of the Rag system.
- `multimodal-drug-detector\ drug_abuse_detector_api.py`: The prediction model logic.
- `multimodal-drug-detector\xai.py`: Contains the XAI component added to the predictions.
- `dataset\`: Contains the datasets for prediction of drug dependency.
- `psychiatrist_portal\`: Contains the frontend codes of the reactjs portal created.
- `server\`: Contains the Flask server setup and setup for all the chains.
- `uploads\`: Folder to contain received images and audios.
- `Dockerfile`: For each code that needs dockerization, there is a dockerfile.
- `requirements.txt`: Under each dockerized code there is a requirement text that goes with it .

## Setup Instructions
Please follow the instructions as they are to get a proper setup
# 🚀 Deployment Instructions (Azure-based)

### 1. Clone the Repository

```bash
git clone https://github.com/olasadek/PsyProtection.git
cd PsyProtection
```

---

### 2. Login to Azure and Azure Container Registry (ACR)

```bash
az login
az acr login --name <your-acr-name>
```

> 📝 Replace `<your-acr-name>` with your actual Azure Container Registry name (e.g., `myregistry`).

---

### 3. Pull Docker Images from ACR

```bash
docker pull <your-acr-name>.azurecr.io/<your-image-name>:<tag>
```

**Example:**

```bash
docker pull myregistry.azurecr.io/drug-detector-api:latest
```

> Repeat for each service image your project uses.

---

### 4. Run the Containers

```bash
docker run -d -p 5000:5000 myregistry.azurecr.io/drug-detector-api:latest
```

> 📌 Adjust the ports (`-p host:container`) depending on the service (e.g., `5000` for API, `5001` for heatmap, etc.)

You can also use `--env` or `--env-file` for any required environment variables:

```bash
docker run -d -p 9000:9000 --env-file .env myregistry.azurecr.io/eep-server:latest
```

---

### 5. Verify Services

Visit the relevant URLs to check if services are up and running:

- `http://<your-vm-ip>:5000` — Prediction API  
- `http://<your-vm-ip>:5001/explain` — Heatmap API  
- `http://<your-vm-ip>:9000` — EEP entrypoint  
- `http://<your-vm-ip>:3000` — UI Portal
