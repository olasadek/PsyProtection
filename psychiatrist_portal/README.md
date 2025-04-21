#  Psychiatric Care: Drug Abuse Detection Platform

Psychiatric Care is a diagnostic platform that uses MRI and Eelectornic health Record data to predict potential drug abuse. The system includes a React-based frontend and multiple Flask APIs for prediction, explanation, and RAG-based treatment recommendation.

---

## 📁 Project Structure

```
psyprotection/
├── psychiatrist-portal/       # React frontend
├── backend/
│   ├── drug_abuse_detector.api  # Prediction API (port 5000)
│   ├── xai_api.py              # Heatmap Explanation API (port 5001)
│   ├── rag_api.py              # RAG Treatment API (port 8000)
│   └── eep_server.py           # Orchestrator / EEP API (port 9000)
```

---

## ⚙️ Requirements

### 🧩 Frontend
- Node.js >= 18
- npm or yarn

### 🐍 Backend
- Python >= 3.8
- pip
- (Recommended) virtualenv

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/PsyProtection.git
cd PsyProtection
```

---

### 2. Run the Frontend (React)

```bash
cd psychiatrist-portal
npm install
npm start
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

### 3. Run the Flask Backends

> Open **four terminals** (or use split terminal tabs):

#### a) Prediction API (port 5000)

```bash
cd backend
pip install -r requirements.txt
python drug_abuse_detector.api
```

#### b) Explanation API (port 5001)

```bash
cd backend
python xai_api.py
```

#### c) RAG API (port 8000)

```bash
cd backend
python rag_api.py
```

#### d) Orchestrator / EEP (port 9000)

```bash
cd backend
python eep_server.py
```

---

## 🧪 Usage

1. In the browser UI, fill out the patient EHR form.
2. Upload a `.nii` or `.nii.gz` MRI scan.
3. Click **Submit**.
4. If prediction is **Drug Abuser**, a heatmap will be generated.
5. RAG will fetch biomedical context and suggest a treatment.

---

## 🛠️ Development Notes

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

You can use the following scripts inside the `psychiatrist-portal` directory:

### Start Frontend

```bash
npm start
```

### Run Tests

```bash
npm test
```

### Build for Production

```bash
npm run build
```

---

## 📄 .gitignore Suggestions

Make sure your `.gitignore` includes:

```
node_modules/
__pycache__/
*.nii
*.nii.gz
*.pt
venv/
.env
build/
```

## 🪪 License

MIT — free to use, modify, and distribute.
