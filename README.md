# AI-Based Lung Cancer Detection Using Medical Image Processing

Educational starter project: image preprocessing + CNN training + Streamlit upload interface.

**Important:** The app is runnable, but predictions require a model trained on the dataset. This repository does not include trained weights or medical data. It is not a medical diagnostic tool.

## Recommended environment
Use **Python 3.11** (not Python 3.14 for this TensorFlow setup).

### 1. Create environment (Windows PowerShell / CMD)
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download dataset
For a beginner-friendly image-classification workflow, use the **IQ-OTH/NCCD Lung Cancer Dataset** from its official Mendeley Data page or the Kaggle listing linked from the dataset page. Check the dataset terms and cite the source in your report.
For CT research with radiologist annotations, use **LIDC-IDRI** from The Cancer Imaging Archive (TCIA). It is much larger and consists primarily of DICOM scans; it needs a separate DICOM/annotation preprocessing pipeline.

Put the extracted image files into this layout (folder names must match):
```text
data/
  train/
    Benign/
    Malignant/
    Normal/
  val/
    Benign/
    Malignant/
    Normal/
  test/
    Benign/
    Malignant/
    Normal/
```
The split should be patient-level where patient identifiers are available. Avoid putting images from the same patient in multiple splits. Do not upload the dataset to GitHub.

### 3. Train
```bat
python train.py
```
The trained model is saved to `models/lung_cancer_model.keras`.

### 4. Run locally
```bat
streamlit run app.py
```
Open the local URL printed in the terminal. Upload an image and view the model output.

### 5. Push to your own GitHub repository
Create an empty repository on GitHub. In this project folder:
```bat
git init
git add .
git commit -m "Initial lung cancer image classification project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```
Replace the URL with your own repo URL. If `origin` already exists, use `git remote set-url origin YOUR-REPO-URL`.

**Do not commit** `.venv`, raw datasets, patient data, or large model files. `.gitignore` excludes these by default. For deployment, use a private/object-storage model artifact or Git LFS only after checking host limits.

## Deployment advice
- **Render (recommended for this starter):** New → Web Service, connect the GitHub repo. Build command: `pip install -r requirements.txt`; Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`. Set Python version to 3.11 if the dashboard offers a version setting. Model loading and package/model size limits still apply.
- **Vercel:** Better for a lightweight frontend/API than a large TensorFlow model. Vercel's Python functions have bundle, memory, and execution limits; a TensorFlow model may not fit or may time out. If you need Vercel, deploy a separate lightweight frontend there and host the model inference API on Render, then configure the frontend to call the Render API. This starter is Streamlit and is not a one-click Vercel deployment.

## Results and responsible use
Report test-set accuracy, precision, recall/sensitivity, specificity, F1-score, and a confusion matrix. Never report sample results as your actual results. Model outputs are experimental predictions, not a diagnosis; consult a qualified clinician for health decisions.
