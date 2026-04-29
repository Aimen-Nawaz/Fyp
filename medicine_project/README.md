# 💊 Medicine Name Reconstruction

Corrects misspelled medicine names using **BiLSTM** neural network + **Levenshtein distance snapping** to real drug names.

## ⚡ Quick Deploy

Deploy to Vercel in 5 minutes using **Git LFS**:  
👉 [**QUICKSTART.md**](QUICKSTART.md)

Model files stay in your GitHub repo—no cloud storage needed!

---

**Produces exactly this output format:**

```
Test Case 1:
Input: panadl
Predicted Output: panadol
Expected Output: panadol ✓

Test Case 2:
Input: asati
Predicted Output: avastin
Expected Output: avastin ✓
```

---

## 🎯 Features

- ✅ **Deep Learning**: BiLSTM model trained on 1,301+ medicine names
- ✅ **Smart Matching**: Combines neural predictions with Levenshtein distance
- ✅ **Real-time Predictions**: Returns alternatives ranked by confidence
- ✅ **Web UI**: Beautiful interactive interface
- ✅ **API**: JSON endpoints for integration
- ✅ **Serverless**: Deployment-ready for Vercel

---

## 📁 Project Structure

```
medicine_project/
├── api/                          ← Vercel serverless functions
│   ├── index.py                  ← Main HTTP handler
│   ├── predict.py                ← Prediction logic
│   └── model_utils.py            ← Model loading
│
├── app/                          ← Legacy Flask app (local testing)
│   └── app.py
│
├── model_artifacts/              ← ML model (git-ignored, ~1GB)
│   ├── medicine_lstm.keras       ← Trained model
│   ├── model_config.json
│   ├── token_to_idx.json
│   ├── idx_to_token.json
│   └── known_names.txt
│
├── notebooks/
│   └── medicine-reconstruction.ipynb ← Training notebook
│
├── data/
│   └── Medecinelist.txt
│
├── vercel.json                   ← Vercel configuration
├── .env.example                  ← Environment variables template
├── .gitignore                    ← Excludes model files
├── requirements.txt              ← Python dependencies
├── QUICKSTART.md                 ← 5-min deployment guide
├── DEPLOYMENT.md                 ← Full deployment guide
└── README.md                     ← This file
```

---

## 🚀 Run Locally

### 1. Setup
```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Option A: Run Flask App
```bash
python app/app.py
# Open http://localhost:5000
```

### 3. Option B: Run Vercel Serverless (Recommended for testing deployment)
```bash
npm install -g vercel
vercel dev
# Open http://localhost:3000
```

---

## 📡 API Endpoints

### **GET / (HTML UI)**
Returns the web interface

### **POST /api/predict**
Single prediction

**Request:**
```json
{
  "input": "panadl"
}
```

**Response:**
```json
{
  "input": "panadl",
  "prediction": "panadol",
  "confidence": 0.94,
  "alternatives": [
    {
      "prediction": "paracetamol",
      "confidence": 0.72
    }
  ]
}
```

### **POST /api/predict_batch**
Batch predictions

**Request:**
```json
{
  "inputs": ["panadl", "asati", "amlodipne"]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "input": "panadl",
      "prediction": "panadol",
      "confidence": 0.94,
      "alternatives": [...]
    },
    ...
  ]
}
```

### **GET /health**
Health check

**Response:**
```json
{
  "status": "ok",
  "model_files": {
    "config": true,
    "t2i": true,
    "i2t": true,
    "model": true,
    "names": true
  }
}
```

---

## 🌐 Deploy to Vercel

### Quick Deploy with Git LFS (5 min)
1. **Install Git LFS:** `git lfs install`
2. **Track models:** `git lfs track "model_artifacts/*"`
3. **Commit & push:** `git push`
4. **Deploy:** `vercel --prod`

### Full Instructions
👉 [**GIT_LFS_DEPLOYMENT.md**](GIT_LFS_DEPLOYMENT.md) - Recommended approach
📖 [**DEPLOYMENT.md**](DEPLOYMENT.md) - Cloud storage alternative

---

## ⚠️ Important: Model Files

The trained model is ~1GB. You have **two deployment options**:

### ✅ **Option 1: Git LFS** (Recommended)
- Keep model files in GitHub using Git LFS
- Deploy directly to Vercel
- No cloud storage needed
- Free (1GB/month on GitHub)
- **👉 [GIT_LFS_DEPLOYMENT.md](GIT_LFS_DEPLOYMENT.md)**

### ✅ **Option 2: Cloud Storage**
- Upload to Google Drive, S3, or GitHub Releases
- Model downloads at runtime
- More complex setup
- **👉 [DEPLOYMENT.md](DEPLOYMENT.md)**

---

## 📁 What's in Your Repo

With Git LFS:
- `model_artifacts/` - **Tracked in Git LFS** (Vercel deploys with it)
- `api/` - Serverless code
- All deployment files - Ready to go!

---

## 🔧 How It Works

```
Input: "panadl"
         ↓
    [Character encoding]
         ↓
[BiLSTM Neural Network] → "panadol" (with confidence)
         ↓
[Levenshtein matching] → Snap to real medicine name
         ↓
Output: "panadol" (0.94 confidence)
```

### Two-Stage Approach
1. **Neural Stage**: LSTM predicts character-by-character
2. **Snap Stage**: Find closest real medicine name using Levenshtein distance

---

## 📊 Model Architecture

| Component | Value |
|-----------|-------|
| **Type** | BiLSTM (Bidirectional LSTM) |
| **Input** | Single medicine name |
| **Output** | Corrected name |
| **Training Data** | 1,301 real medicine names |
| **Loss Function** | Masked Sparse Categorical Crossentropy |
| **Optimizer** | Adam (lr=0.001) |
| **Framework** | TensorFlow/Keras |

---

## 📦 Dependencies

```
tensorflow==2.14.0
numpy==1.24.3
python-Levenshtein==0.21.1
flask==3.0.0
flask-cors==4.0.0
```

---

## 🔍 Troubleshooting

### **Model files not found locally**
- Download from Kaggle and place in `model_artifacts/`
- Or adjust `ARTIFACTS_DIR` in code

### **Import errors (TensorFlow, etc.)**
```bash
pip install --upgrade -r requirements.txt
```

### **Vercel deployment issues**
- Check `/health` endpoint
- Review logs: `vercel logs --tail`
- Test locally: `vercel dev`

---

## 📝 License

[Add your license here]

---

## 👨‍💻 Author

Medicine Name Reconstruction Project

**Project Files:**
- 🚀 Quick Deploy: [QUICKSTART.md](QUICKSTART.md)
- 📖 Full Guide: [DEPLOYMENT.md](DEPLOYMENT.md)
- 📓 Training: [notebooks/medicine-reconstruction.ipynb](notebooks/medicine-reconstruction.ipynb)

---

## 🚀 Step-by-Step Kaggle Instructions

### Step 1: Upload the drug list
1. Go to **kaggle.com** → **Datasets** → **New Dataset**
2. Upload `data/generic_drugs.txt`
3. Give it a slug like `my-medicines` — note the slug!

### Step 2: Upload the notebook
1. **Code** → **New Notebook** → **File → Import Notebook**
2. Upload `notebooks/Medicine_Reconstruction.ipynb`

### Step 3: Configure
1. Click **Add Data** → find your uploaded dataset → **Add**
2. On the right you'll see path like `/kaggle/input/my-medicines/generic_drugs.txt`
3. In the notebook **Step 1 Config cell**, paste that exact path:
   ```python
   DRUGS_LIST = '/kaggle/input/YOUR-SLUG/generic_drugs.txt'
   ```
4. Settings → **Accelerator → GPU T4 x2**

### Step 4: Run All
- Click **Run All**
- Training takes ~15 min on GPU
- Watch **Step 10** output — it verifies all 5 files exist

### Step 5: Download files from Kaggle Output
After running, look at the **Output** panel on the right. You should see:
```
medicine_lstm.keras       ✅
token_to_idx.json         ✅
idx_to_token.json         ✅
model_config.json         ✅
known_names.txt           ✅
medicine_pairs.csv
training_curves.png
```

Download all 5 of the first files (the top ones with ✅).

**If files don't appear in Output:** 
- Check **Step 1** config uses `/kaggle/working/` paths (not bare filenames)
- Re-run **Step 10** explicitly — it saves everything again

### Step 6: Run Locally
```bash
# Put downloaded 5 files into model_artifacts/
pip install -r app/requirements_app.txt
python app/app.py
# Open http://localhost:5000
```

---

## 📊 Expected Results Match Reference Image

| Input | Predicted | Expected | Match |
|-------|-----------|----------|-------|
| panadl | panadol | panadol | ✓ |
| asati | avastin | avastin | ✓ |
| amlodipne | amlodipine | amlodipine | ✓ |
| hydrcodone | hydrocodone | hydrocodone | ✓ |
| morhine | morphine | morphine | ✓ |
