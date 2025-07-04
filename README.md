# **![pink badge](https://img.shields.io/badge/Psychiatric-Care-ffc0cb)**
A web portal to help assist the dependency of psychiatric patients on their perscribed medicines.
Knowing that over 30% of psychiatric patients, including those in psychwards, are heavily prone to becoming addicted to their prescribed medicines, 
we have came with an assistant system solution to help monitor patients and to detect earlier-on their dependency on meds likelihood. 
This project levarages bran MRI and electronic health data to build an AI-driven monitoring system that:
 1. Predicts Addiction Risk
 2. Self explains its outcomes.
 3. Recommends personalized tapering plans.


![mental-health-banner-2200x1200](https://github.com/user-attachments/assets/d0084083-10b3-4125-bf09-a2e55117cc2f)


### The google drive link for all needed Data and models weights :
https://drive.google.com/drive/folders/1v8vINEB3Vt5aWw2HX6OdrRbH0fgPK2iP?usp=drive_link


 # Content table: 

- [Features](#the-features)
- [Datasets used](#datasets-used)
- [Project Structure](#project-structure)
- [The workflow](#the-workflow)
- [Setup Instructions](#setup-instructions)
- [CI CD](#CI/CD)
- [Unit testing](#unit-testing)
- [Meet the team](#meet-the-team)
- [Happy analysis!](#happy-analysis)


# The features

- ![pink badge](https://img.shields.io/badge/Multimodal-model-ffc0cb): 
  - Processes **Electronic health record data** (Dense Neural Network) and **3D MRI volume images** (3D Vision Transformers).
  - Combines features using a **Late Fusion Classifier**.
  - **Training Accuracy**: 97%, **Testing Accuracy**: 87%.

- ![pink badge](https://img.shields.io/badge/Explainable-AI-ffc0cb):
  - A sliding window with the model's weight gained from the fine tuned ViT model slides over the images to find abnormalities.
  - Generates a **heatmap** using occlusion-based Grad-CAM analysis to highlight important brain areas related to addiction risk.
  - Provides transparency by showing which parts of the MRI contribute to the model's prediction providing an extensive explanation of abnormailities in the brain.

- ![pink badge](https://img.shields.io/badge/RAG-system-ffc0cb):
  - Retrieves research articles from **PMC** and **ArXiv** based on user queries.
  - Articles are chunked, embedded, and stored in a **vector database**.
  - Uses a **GPT model** to rewrite queries and generate answers based on the research.
  - **Cached in TinyDB** for fast responses to repeated queries.
**Automated Pipeline is implemented if the prediction of the patient is drug abuser where the explainable AI component will give back
   a heatmap and a query will form using his medication and illness and thrown into the rag system for recommendation treatment**

# Datasets used:

This project utilizes the following datasets:

1.  ![pink badge](https://img.shields.io/badge/Text-Image-ffc0cb)
   - Description: A wide collection of  3d-volume files of Brain MRI with the corresponding health data.
   - Source: OpenNeuro addiction dataset.
   - Usage: Used to train the multimodal prediction model.

2. ![pink badge](https://img.shields.io/badge/Image-data-ffc0cb)
   - Description: Contains text data from articles .
   - Source:  from both pubmed and arxiv via retrieval tools.
   - Usage: Utilized for augmentation and answer generation for the rag system.

# Project Structure

- `EEP\`: contains the eep server code that orchastrates the ports.
- `rag.py\`: Contains the logic of the Rag system.
- `multimodal-drug-detector\ drug_abuse_detector_api.py`: The prediction model logic.
- `multimodal-drug-detector\xai.py`: Contains the XAI component added to the predictions.
- `dataset\`: Contains the datasets for prediction of drug dependency.
- `psychiatrist_portal\`: Contains the frontend codes of the reactjs portal created.
- `k8`\: For running this project with kubernetes and the corresponding pods
- `Dockerfile`: For each code that needs dockerization, there is a dockerfile.
- `requirements.txt`: Under each dockerized code there is a requirement text that goes with it .


# The workflow 

![deepseek_mermaid_20250426_03bd15](https://github.com/user-attachments/assets/24f7395c-8def-43e7-9ef8-070475607d4c)

#  Setup Instructions
Please follow the instructions as they are to get a proper setup

### 1. Clone the Repository 
```bash
git clone https://github.com/olasadek/PsyProtection.git
cd PsyProtection
```
please never forget to change the environment file / configure file to match your api key and entrez email

## First choice with ![pink badge](https://img.shields.io/badge/via-kubernetes-ffc0cb)  :

Requires:
Open AI api key
Entrez email
azure servers names and passwords
kubectl installed and configured
Docker images already pushed to your registry



## 2. Deploy all services
```bash
kubectl apply -f portal_server.yaml \
              -f eep_server.yaml \
              -f drug_detector.yaml \
              -f xai.yaml \
              -f rag_server.yaml
```

## 3. Get access URLs (run in separate terminal)
```bash
minikube tunnel  # For Minikube users
kubectl get svc portal-server eep-server -w
```
## 4. Deployment Commands
```bash
kubectl apply -f portal_server.yaml
kubectl apply -f eep_server.yaml
kubectl apply -f drug_detector.yaml
kubectl apply -f xai.yaml
kubectl apply -f rag_server.yaml
# Verify deployment
kubectl get pods -w
```
## 5. Access Your Application
```bash
kubectl get svc -w
```
## 6. Access URLs

**Web Portal:** http://203.0.113.10:3000

**EEP API:** http://203.0.113.11:9000 

## 7. Shutdown when done.
```bash
kubectl delete -f portal_server.yaml
kubectl delete -f eep_server.yaml
kubectl delete -f drug_detector.yaml
kubectl delete -f xai.yaml
kubectl delete -f rag_server.yaml
```

## Second choice via ![pink badge](https://img.shields.io/badge/docker-compose-ffc0cb)
Requires:
Open AI api key
Entrez email
azure servers names and passwords


### 1. Pull Docker Images from ACR

```bash
docker pull absue.azurecr.io/abuse-drug-detector-api:latest
```

> Repeat for each service image .
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
- `http://<your-vm-ip>:8000/ask_question` — Rag API
  
## Additional if you wish to try it without ![pink badge](https://img.shields.io/badge/using-docker-ffc0cb) if for some reason you wanted an easy run:

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

**ATTENTION : don't worry if you do not have an openai api key, you can still go for a normal analysis and have a good explanation for the outcomes regardless.**
**you will just lose the recommendation priviliges**

# Outcome:
### you should be able to see a dashboard asking you for electronic health data and a brain mri file like this:

<img width="931" alt="preview" src="https://github.com/user-attachments/assets/d745cce1-ebd4-4263-a36a-566b14ca0688" />

### If you correctly inputed everything you should be getting a complete analysis of your corresponding patient : 

![Picture1](https://github.com/user-attachments/assets/9b241ada-36d5-4d78-9e2c-9777dba66970)

# CI/CD:
By using azure containers registry we assured the continuous integration and deployment and the monitoring of requests coming in dockerized containers.

<img width="434" alt="Screenshot 2025-04-22 111821" src="https://github.com/user-attachments/assets/3497883d-d618-4a26-8147-39f16632c5b8" />

# Unit testing:
 every single part of this project has been subjected to UnitTest from local docker containers to azure deployed containers to docker-compose using Postman

# Happy analysis and may your patients be always safe !
In Hope of helping psychiatrists and psychwards better treat and monitor their patients.

## Meet the team:
### Ola Sadek - masters student @ american university of beirut.
### Oussama Ibrahim - masters student @ american university of beirut.

we would love to hear your feedback .

