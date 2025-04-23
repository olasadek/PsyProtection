![pink badge](https://img.shields.io/badge/Psychiatric-Care-ffc0cb)  
# Psychiatric Care

A web portal to help assist the dependency of psychiatric patients on their perscribed medicines.


![mental-health-banner-2200x1200](https://github.com/user-attachments/assets/d0084083-10b3-4125-bf09-a2e55117cc2f)


# The google Drive for all needed Data and models weights : 
https://drive.google.com/drive/folders/1v8vINEB3Vt5aWw2HX6OdrRbH0fgPK2iP?usp=drive_link


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

1. **Electronic health records and MRI Dataset**
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
- `Dockerfile`: For each code that needs dockerization, there is a dockerfile.
- `requirements.txt`: Under each dockerized code there is a requirement text that goes with it .

## Setup Instructions
Please follow the instructions as they are to get a proper setup
# 🚀 Deployment Instructions (Azure-based)
Requires:
Open AI api key
Entrez email
azure servers names and passwords

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


---

### 3. Pull Docker Images from ACR

```bash
docker pull absue.azurecr.io/abuse-drug-detector-api:latest
```

> Repeat for each service image .

---

### 4. Run the Containers

```bash
docker run -d -p 5000:5000 absue.azurecr.io/abuse-drug-detector-api:latest
```

> 📌 Adjust the ports (`-p host:container`) depending on the service (e.g., `5000` for /drug_abuse_detector, `5001` for XAI, etc.)

You can also use `--env` or `--env-file` for any required environment variables:

```bash
docker run -d -p 9000:9000 --env-file .env myregistry.azurecr.io/eep-server:latest
```

---

### 5. Verify Services

Visit the relevant URLs to check if services are up and running:

- `http://<your-vm-ip>:5000/drug_abuse_detector` — Prediction API  
- `http://<your-vm-ip>:5001/explain` — Heatmap API  
- `http://<your-vm-ip>:9000/analyze_patient` — EEP entrypoint  
- `http://<your-vm-ip>:3000` — UI Portal
- - `http://<your-vm-ip>:8000/ask_question` — Rag API
## Additional if you wish to try it without the docker setup:

### 1. Clone the Repository

```bash
git clone https://github.com/olasadek/PsyProtection.git
cd PsyProtection
```

### 2. Have 5 terminals ready for this set up

```bash
python xai_api.py
cd PsyProtection\multimodal-drug-detector
```
repeat for drug_abuse_detector_api.py // rag.py (don't forget to configure your environment in the environment file in the same folder // eep_server.py // xai_api.py . 
KEEP THEM ALL RUNNING FOR THIS TO WORK

### 3. Turn on the portal 

for the last terminal 
```bash
npm start

```
you will be directed to local port 3000 where enginex usually runs. 
Enter your data and analyze!

### ATTENTION : don't worry if you do not have an openai api key, you can still go for a normal analysis and have a good explanation for the outcomes regardless.
### you will just lose the recommendation priviliges

## Happy analysis and may your patients be always safe !
In Hope of helping psychiatrists and psychwards better treat and monitor their patients.


