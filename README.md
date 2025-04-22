# Psychiatric Care

A web portal to help assist the dependency of psychiatric patients on their perscribed medicines

<img src="frontend/images/symptom_disease_icon.png" alt="Symptom Disease Icon" width="200" height="170" />

## Table of Contents

- [Features](#features)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Configuration and Running](#configuration-and-running)
- [Meet the Team](#meet-the-team)
- [Acknowledgements](#acknowledgements)

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
