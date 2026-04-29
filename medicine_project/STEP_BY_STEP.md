# 📋 Detailed Step-by-Step Deployment Instructions

## **Overview**
This document provides **complete step-by-step instructions** for deploying the Medicine Name Reconstruction app to Vercel.

---

## **Stage 1: Prepare Model Files**

### **Step 1.1: Zip Model Artifacts**
```bash
# Navigate to project root
cd c:\Users\hp\Downloads\medicine_project (1)\medicine_project

# Create zip file (Windows PowerShell)
Compress-Archive -Path .\model_artifacts -DestinationPath model_artifacts.zip

# Or (Windows CMD)
tar.exe -a -c -f model_artifacts.zip model_artifacts/

# Or (macOS/Linux)
zip -r model_artifacts.zip model_artifacts/
```

### **Step 1.2: Upload to Cloud Storage**

#### **Option A: Google Drive** (Recommended - Easiest)

1. **Upload file:**
   - Go to https://drive.google.com
   - Create folder named `medicine-model`
   - Upload `model_artifacts.zip`

2. **Get shareable link:**
   - Right-click file → Share
   - Click "Change to anyone with the link"
   - Copy link: `https://drive.google.com/file/d/{FILE_ID}/view`

3. **Create download URL:**
   - Extract FILE_ID from link
   - Create direct download: `https://drive.google.com/uc?id={FILE_ID}&export=download`
   - **Keep this URL for later**

**Example:**
```
Share link: https://drive.google.com/file/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ/view
FILE_ID: 1aBcDeFgHiJkLmNoPqRsTuVwXyZ
Download URL: https://drive.google.com/uc?id=1aBcDeFgHiJkLmNoPqRsTuVwXyZ&export=download
```

#### **Option B: AWS S3**

1. **Create S3 bucket:**
   ```bash
   aws s3 mb s3://medicine-model-bucket
   ```

2. **Upload file:**
   ```bash
   aws s3 cp model_artifacts.zip s3://medicine-model-bucket/
   ```

3. **Make public:**
   - Go to AWS S3 console
   - Right-click file → Make public
   - Copy URL: `https://medicine-model-bucket.s3.amazonaws.com/model_artifacts.zip`
   - **Keep this URL for later**

#### **Option C: GitHub Releases**

1. **Create release:**
   - Go to GitHub repo → Releases → New Release
   - Tag: `v1.0-models`
   - Attach `model_artifacts.zip`
   - Publish

2. **Get download URL:**
   - Copy from Assets: `https://github.com/USERNAME/REPO/releases/download/v1.0-models/model_artifacts.zip`
   - **Keep this URL for later**

---

## **Stage 2: Prepare Git Repository**

### **Step 2.1: Verify Git Setup**
```bash
cd medicine_project

# Check git status
git status

# Add all deployment files
git add vercel.json api/ DEPLOYMENT.md QUICKSTART.md .gitignore

# Verify model_artifacts is ignored
git status | grep model_artifacts  # Should be empty

# Commit
git commit -m "feat: add Vercel serverless deployment configuration"
```

### **Step 2.2: Push to GitHub**
```bash
# If not already pushed
git push -u origin main

# Or if already pushed
git push
```

---

## **Stage 3: Install Vercel CLI**

### **Step 3.1: Install**
```bash
# Using npm (if you have Node.js)
npm install -g vercel

# Or using yarn
yarn global add vercel

# Verify installation
vercel --version  # Should show version number
```

**If Node.js not installed:**
1. Download from https://nodejs.org/
2. Install
3. Run npm command above

### **Step 3.2: Login**
```bash
# Login to Vercel
vercel login

# Browser will open, authorize your GitHub/GitLab/Bitbucket account
# Confirm login in terminal
```

---

## **Stage 4: Deploy to Vercel**

### **Step 4.1: Initial Deployment**
```bash
# From project root
cd medicine_project

# Deploy (select yes to link to git)
vercel

# Answer prompts:
# Set up and deploy? › yes
# Which scope do you want to deploy to? › Your-Username
# Link to existing project? › no (for first time)
# What's your project's name? › medicine-name-reconstruction
# In which directory is your code? › ./
# Want to modify vercel.json? › no
```

**Result:**
```
✅ Production: https://medicine-name-reconstruction.vercel.app
✅ Preview: https://medicine-name-reconstruction-git-main.vercel.app
```

### **Step 4.2: Verify Deployment (Initial Check)**
```bash
# Check if deployed
curl https://medicine-name-reconstruction.vercel.app/

# Should return HTML (the web UI)
```

---

## **Stage 5: Configure Environment Variables**

### **Step 5.1: Set Variables in Vercel Dashboard**

1. **Open Vercel Dashboard:**
   - Go to https://vercel.com/dashboard

2. **Select your project:**
   - Click `medicine-name-reconstruction`

3. **Go to Settings:**
   - Click **Settings** tab (top menu)
   - Click **Environment Variables** (left sidebar)

4. **Add Variables:**

   **Variable 1:**
   ```
   Name: MODEL_LOADING_MODE
   Value: download
   Environments: All (Production, Preview, Development)
   ```
   Click **Add**

   **Variable 2:**
   ```
   Name: MODEL_DOWNLOAD_URL
   Value: https://drive.google.com/uc?id=YOUR_FILE_ID&export=download
         (or your S3/GitHub URL from Stage 1.2)
   Environments: All (Production, Preview, Development)
   ```
   Click **Add**

5. **Redeploy:**
   - Go to **Deployments** tab
   - Click "..." on latest deployment
   - Click **Redeploy**
   - Wait ~2-3 minutes

---

## **Stage 6: Verify Deployment**

### **Step 6.1: Check Health Endpoint**
```bash
# Test health endpoint
curl https://medicine-name-reconstruction.vercel.app/health

# Expected response (all files should be true after download):
{
  "status": "ok",
  "model_files": {
    "config": true,
    "i2t": true,
    "model": true,
    "names": true,
    "t2i": true
  }
}
```

### **Step 6.2: Test Single Prediction**
```bash
curl -X POST https://medicine-name-reconstruction.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input":"panadl"}'

# Expected response:
{
  "input": "panadl",
  "prediction": "panadol",
  "confidence": 0.94,
  "alternatives": [...]
}
```

### **Step 6.3: Test Batch Prediction**
```bash
curl -X POST https://medicine-name-reconstruction.vercel.app/api/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"inputs":["panadl","asati","amlodipne"]}'

# Expected response with 3 predictions
```

### **Step 6.4: Open Web UI**
```
https://medicine-name-reconstruction.vercel.app
```
- Should see interactive web interface
- Try predicting a medicine name

---

## **Stage 7: Monitor Deployment**

### **Step 7.1: View Logs**
```bash
# Real-time logs
vercel logs --tail

# Filter for errors
vercel logs | grep ERROR
```

### **Step 7.2: Check Deployment Status**
```bash
# List all deployments
vercel list

# Check specific deployment
vercel inspect <deployment-url>
```

### **Step 7.3: Common Issues & Fixes**

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| **404 on /health** | App not deployed | `vercel deploy --prod` again |
| **Model files not found** | Download failed | Verify `MODEL_DOWNLOAD_URL` in env vars |
| **500 error on prediction** | Model loading error | Check `vercel logs --tail` for details |
| **Timeout (>30s)** | Cold start too slow | First request is slow (TensorFlow loads), wait 2-3 min |

---

## **Stage 8: Production Configuration (Optional)**

### **Step 8.1: Custom Domain**
1. Buy domain (e.g., medicine-reconstructor.com)
2. Vercel Dashboard → Settings → Domains
3. Add custom domain
4. Follow DNS instructions

### **Step 8.2: Enable Auto-Deployments from Git**
1. Vercel Dashboard → Git Integration → Connected
2. Auto-deploy on push to main branch

### **Step 8.3: Create Team/Collaborators**
1. Vercel Dashboard → Settings → Members
2. Invite teammates

---

## **Complete Checklist**

- [ ] Model artifacts zipped
- [ ] Uploaded to cloud storage (Google Drive/S3/GitHub)
- [ ] Download URL tested and working
- [ ] Git repository initialized with deployment files
- [ ] All files committed and pushed
- [ ] Node.js installed
- [ ] Vercel CLI installed (`vercel --version`)
- [ ] Logged into Vercel (`vercel login`)
- [ ] Initial deployment done (`vercel`)
- [ ] Environment variables set in Vercel dashboard
- [ ] Redeployed after env var changes
- [ ] `/health` endpoint returns all files as `true`
- [ ] `/api/predict` endpoint works correctly
- [ ] Web UI loads and is interactive
- [ ] Logs checked for errors (`vercel logs`)

---

## **Troubleshooting Flowchart**

```
App deployed but not working?
│
├─ Check /health endpoint
│  ├─ All files true? → Try prediction
│  │  ├─ Works? → ✅ Success!
│  │  └─ Error? → Check logs (vercel logs --tail)
│  └─ Files false? → MODEL_DOWNLOAD_URL incorrect
│     └─ Fix URL in Vercel Settings → Redeploy
│
└─ 404 error?
   └─ Vercel deploy failed
      └─ Run: vercel deploy --prod
```

---

## **Getting Help**

1. **Check Vercel Documentation:**
   - https://vercel.com/docs/concepts/functions/serverless-functions/python

2. **Check Function Logs:**
   ```bash
   vercel logs --tail
   ```

3. **Test Locally First:**
   ```bash
   vercel dev  # Runs locally on http://localhost:3000
   ```

4. **Verify URLs:**
   - Model download: Test in browser (should start download)
   - API endpoint: Test with curl (examples above)

5. **Check File Existence:**
   ```bash
   curl https://your-project.vercel.app/health
   ```

---

## **Next Steps After Deployment**

1. ✅ **Share with others:**
   - Send link: `https://your-project.vercel.app`
   - Share API endpoint for integration

2. ✅ **Monitor usage:**
   - Check logs regularly
   - Monitor performance in Vercel dashboard

3. ✅ **Update model (if needed):**
   - Re-zip and upload new model
   - Update `MODEL_DOWNLOAD_URL` in Vercel
   - Redeploy

4. ✅ **Set up CI/CD (optional):**
   - Auto-deploy on git push
   - Run tests before deployment

---

## **File Reference**

- **vercel.json** - Vercel configuration
- **api/index.py** - Main handler
- **api/predict.py** - Prediction logic
- **api/model_utils.py** - Model loading
- **.env.example** - Environment template
- **.gitignore** - Excludes model files

**All ready!** Your app is now deployment-ready. 🚀
