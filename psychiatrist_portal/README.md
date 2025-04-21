# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.
# 🧠 PsyProtection: Drug Abuse Detection Platform

This is a multimodal diagnostic tool for detecting potential drug abuse based on MRI and EHR data. The system consists of a React frontend and Flask-based backend with multiple services.

---

## 📁 Project Structure

```
psyprotection/
├── psychiatrist-portal/         # Frontend (React)
├── api_predict/                 # Flask backend (port 5000)
├── api_heatmap/                 # Flask XAI service (port 5001)
├── orchestrator/                # Entry Endpoint Processor (port 9000)
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

### 1. Clone the repository

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

The app will start at: [http://localhost:3000](http://localhost:3000)

---

### 3. Run the Backend APIs

> Open **three terminals** (or split terminals in VS Code):

#### a) Prediction API (port 5000)

```bash
cd api_predict
pip install -r requirements.txt
python api_predict.py
```

#### b) Explanation API (port 5001)

```bash
cd api_heatmap
pip install -r requirements.txt
python api_heatmap.py
```

#### c) Orchestrator API (port 9000)

```bash
cd orchestrator
pip install -r requirements.txt
python orchestrator.py
```

---

## 🧪 Usage

1. Fill out the patient EHR form in the UI.
2. Upload a `.nii` or `.nii.gz` MRI file.
3. Submit the form.
4. If the result is **"Drug Abuser"**, a heatmap explanation is displayed.
5. A treatment recommendation is shown using biomedical RAG.

---

## 🛠️ Ignore These in Git

Add the following to your `.gitignore`:

```
node_modules/
__pycache__/
.env
*.nii
*.nii.gz
*.pt
venv/
```

