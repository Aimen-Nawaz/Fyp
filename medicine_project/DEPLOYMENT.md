# 🚀 Vercel Deployment Guide - Medicine Name Reconstruction

## **Overview**
This guide explains how to deploy the Medicine Name Reconstruction app to Vercel.

---

## **⚠️ Critical: Model Files**

The ML model artifacts are **too large for Vercel** (default 50MB limit). You have **3 options**:

### **Option A: Local Development Only** (Recommended for Testing)
- Keep `model_artifacts/` folder with your project
- Deploy to Vercel WITHOUT model files
- Add model download at startup (see below)

### **Option B: Download from Kaggle at Runtime** (Recommended for Production)
- Host model file online (Google Drive, AWS S3, etc.)
- Download on first request
- Add to [.env.example](.env.example):
  ```
  MODEL_LOADING_MODE=download
  MODEL_DOWNLOAD_URL=https://your-storage-url/model_artifacts.zip
  ```

### **Option C: Store in Vercel KV Storage**
- Use Vercel's KV database
- Requires Vercel Pro ($20/month)
- More complex setup

---

## **📋 Pre-Deployment Checklist**

- [ ] Python 3.11+ installed
- [ ] Git repository initialized
- [ ] `.gitignore` updated (model files excluded)
- [ ] `requirements.txt` current
- [ ] `vercel.json` in root
- [ ] `api/` folder with all modules
- [ ] `.env.example` created

---

## **🔧 Step 1: Setup Locally**

### 1.1 Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 1.2 Test Locally (with model artifacts)
```bash
# Make sure model_artifacts/ folder exists in project root
# Then run the Flask app for testing:
python app/app.py

# Or test the serverless handler:
vercel dev
```

---

## **🌐 Step 2: Deploy to Vercel**

### 2.1 Install Vercel CLI
```bash
npm install -g vercel
# or
yarn global add vercel
```

### 2.2 Login to Vercel
```bash
vercel login
```

### 2.3 Deploy
```bash
# First deployment (creates project)
vercel

# Subsequent deployments
vercel deploy

# Production deployment
vercel deploy --prod
```

---

## **🔌 Step 3: Handle Model Files**

### **For Development (Local Testing)**
1. Keep `model_artifacts/` in project root
2. Add to `.gitignore` ✓ (already done)
3. Model loads from local folder in development

### **For Production (Vercel)**

#### **Option B.1: Download from Google Drive**
1. Upload `model_artifacts.zip` to Google Drive
2. Get shareable link: `https://drive.google.com/file/d/{FILE_ID}/view`
3. Create direct download URL: `https://drive.google.com/uc?id={FILE_ID}&export=download`
4. In Vercel dashboard → Project → Settings → Environment Variables:
   ```
   MODEL_LOADING_MODE = download
   MODEL_DOWNLOAD_URL = https://drive.google.com/uc?id=YOUR_FILE_ID&export=download
   ```

#### **Option B.2: Download from AWS S3**
1. Upload `model_artifacts.zip` to S3
2. Make it public or use signed URL
3. Set environment variable:
   ```
   MODEL_LOADING_MODE = download
   MODEL_DOWNLOAD_URL = https://your-bucket.s3.amazonaws.com/model_artifacts.zip
   ```

#### **Option B.3: Download from GitHub Releases**
1. Create GitHub release with `model_artifacts.zip`
2. Get download URL: `https://github.com/{owner}/{repo}/releases/download/{tag}/model_artifacts.zip`
3. Set environment variable with GitHub token if private

---

## **🔐 Environment Variables**

Set in Vercel Dashboard → Project → Settings → Environment Variables:

```bash
MODEL_LOADING_MODE=download              # or 'local'
MODEL_DOWNLOAD_URL=https://...           # Your cloud storage URL
PYTHONUNBUFFERED=1
```

Or create `.env.local` for local testing:
```bash
cp .env.example .env.local
# Edit .env.local with your values
```

---

## **✅ Verify Deployment**

### **Check Health Endpoint**
```bash
curl https://your-project.vercel.app/health
```

**Expected response:**
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

### **Test Prediction**
```bash
curl -X POST https://your-project.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "panadl"}'
```

**Expected response:**
```json
{
  "input": "panadl",
  "prediction": "panadol",
  "confidence": 0.95,
  "alternatives": [...]
}
```

---

## **🐛 Troubleshooting**

### **"Model files not found"**
**Cause:** Model artifacts not included in deployment
**Fix:**
- If using Option A: Add to Vercel (may exceed limits)
- If using Option B: Set `MODEL_LOADING_MODE=download` and `MODEL_DOWNLOAD_URL`
- Check `/health` endpoint for file status

### **"500 Internal Server Error"**
**Cause:** Model loading or prediction failure
**Fix:**
1. Check Vercel function logs: `vercel logs`
2. Verify environment variables are set
3. Test locally: `vercel dev`

### **"Cold start timeout"**
**Cause:** Model too large, takes >60s to load
**Fix:**
- Increase `maxDuration` in `vercel.json` (max 300s on Pro)
- Pre-warm with health check: `curl https://your-project.vercel.app/health`

### **CORS Errors**
**Fix:** Already handled in `api/index.py` with `Access-Control-Allow-Origin: *`

---

## **📦 Project Structure for Deployment**

```
medicine_project/
├── api/                          ← Vercel serverless functions
│   ├── index.py                  ← Main handler
│   ├── predict.py                ← Prediction logic
│   ├── model_utils.py            ← Model loading
│   └── __init__.py               (optional, for Python package)
│
├── model_artifacts/              ← EXCLUDED from git (.gitignore)
│   ├── model_config.json         (only in development)
│   ├── medicine_lstm.keras       (only in development)
│   ├── token_to_idx.json         (only in development)
│   ├── idx_to_token.json         (only in development)
│   └── known_names.txt           (only in development)
│
├── app/                          ← Legacy Flask (optional)
│   ├── app.py                    (for local testing)
│   └── requirements_app.txt
│
├── .env.example                  ← Template for env vars
├── .gitignore                    ← Excludes model files
├── vercel.json                   ← Vercel configuration
├── requirements.txt              ← Python dependencies
└── README.md
```

---

## **🚦 Deployment Checklist - Final**

- [ ] `api/` folder contains all Python files
- [ ] `vercel.json` configured correctly
- [ ] `.gitignore` includes `model_artifacts/`
- [ ] `requirements.txt` has correct versions
- [ ] Environment variables set in Vercel dashboard
- [ ] Model download URL is accessible (if using Option B)
- [ ] Tested locally with `vercel dev`
- [ ] Pushed to git with `git push`
- [ ] Deployed with `vercel --prod`
- [ ] Verified `/health` endpoint returns success
- [ ] Tested prediction endpoint works

---

## **📞 Getting Help**

1. **Check Vercel Logs:**
   ```bash
   vercel logs --tail
   ```

2. **Test Locally First:**
   ```bash
   vercel dev
   # Then test at http://localhost:3000
   ```

3. **Verify Model Files:**
   ```bash
   curl https://your-project.vercel.app/health
   ```

4. **Vercel Documentation:**
   - https://vercel.com/docs/concepts/functions/serverless-functions
   - https://vercel.com/docs/concepts/functions/serverless-functions/python

---

## **🎯 Next Steps**

1. Upload model artifacts to cloud storage (Google Drive, S3, etc.)
2. Set environment variables in Vercel dashboard
3. Run `vercel --prod` to deploy
4. Test the live endpoint
5. Monitor logs with `vercel logs --tail`
