# 🚀 Deploy with Git LFS - No Cloud Storage Needed

This approach keeps model files in your Git repository using **Git LFS** (Large File Storage). Files are deployed directly to Vercel.

---

## **How It Works**

```
Your Local Machine (model_artifacts/)
         ↓
   (Git LFS tracks)
         ↓
    GitHub LFS
         ↓
Vercel deploys (with model files)
         ↓
Live app with model
```

**Benefits:**
- ✅ No cloud storage needed (Google Drive, S3, etc.)
- ✅ Model files stay in your repo
- ✅ Automatic deployment to Vercel
- ✅ Simple setup (5 minutes)

---

## **⚠️ Important Limitation**

**Free GitHub accounts:** 1GB free LFS bandwidth/month
- If model files > 1GB total, you may need GitHub Pro ($4/month) for more LFS bandwidth

**Check your model size:**
```bash
dir model_artifacts\  # Windows
# or
ls -lh model_artifacts/  # macOS/Linux
```

---

## **Setup Steps**

### **Step 1: Install Git LFS** (One-time)

#### **Windows**
```bash
# Download installer
choco install git-lfs
# or download from https://git-lfs.github.com

# Verify installation
git lfs version  # Should show version
```

#### **macOS**
```bash
brew install git-lfs
git lfs install
```

#### **Linux**
```bash
sudo apt install git-lfs
git lfs install
```

**Or download:** https://git-lfs.github.com

---

### **Step 2: Initialize Git LFS in Your Repository**

```bash
cd medicine_project

# One-time setup
git lfs install
```

---

### **Step 3: Track Model Files with Git LFS**

```bash
# Add large files to Git LFS
git lfs track "model_artifacts/*.keras"
git lfs track "model_artifacts/*.json"
git lfs track "model_artifacts/*.txt"

# Or track the entire folder
git lfs track "model_artifacts/*"
```

**This creates `.gitattributes` file** (commit this!)

```bash
# Commit the tracking setup
git add .gitattributes
git commit -m "chore: setup Git LFS for model artifacts"
```

---

### **Step 4: Add Model Files to Git**

```bash
# Remove from .gitignore if it's there
# Edit .gitignore and DELETE the line: "model_artifacts/"

# Add model files
git add model_artifacts/

# Commit
git commit -m "feat: add model artifacts (tracked with Git LFS)"

# Push (this uploads to GitHub LFS)
git push
```

**First push takes a while** (uploading large files)

```bash
# Check status
git lfs ls-files  # Shows all LFS-tracked files
```

---

### **Step 5: Verify GitHub Setup**

1. Go to https://github.com/your-username/medicine_project
2. Click `model_artifacts/` folder
3. You should see files (model_artifacts/ is NO LONGER in .gitignore)

---

### **Step 6: Deploy to Vercel**

```bash
# From project root
vercel --prod

# No additional configuration needed!
# Vercel automatically downloads model files from GitHub LFS
```

**That's it!** Vercel will:
1. Clone your repo (including LFS files)
2. Deploy with model artifacts included
3. Load models locally (no cloud download needed)

---

## **Verification**

### **Check Git LFS Status**
```bash
git lfs ls-files
# Should show all model files

git lfs status
# Shows all tracked files
```

### **Check Vercel Deployment**
```bash
# Test health endpoint
curl https://your-project.vercel.app/health

# Expected: all model files = true
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

### **Test Prediction**
```bash
curl -X POST https://your-project.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input":"panadl"}'

# Expected response:
{
  "input": "panadl",
  "prediction": "panadol",
  "confidence": 0.94,
  ...
}
```

---

## **Troubleshooting**

### **"Model files not found" on Vercel**

**Cause:** Git LFS files not uploaded correctly

**Fix:**
```bash
# Check local Git LFS status
git lfs ls-files

# Push again
git push

# Check GitHub (files should show size in MB, not pointer)
# https://github.com/your-username/medicine_project/blob/main/model_artifacts/

# Redeploy Vercel
vercel --prod
```

### **Vercel deployment too slow**

**Cause:** Vercel is cloning LFS files (normal, takes 2-3 min)

**Fix:** This is expected on first deployment. Subsequent deploys are faster.

### **"Out of Git LFS bandwidth"** (GitHub warning)

**Solution:** Upgrade to GitHub Pro ($4/month) for 1TB LFS bandwidth

---

## **Workflow After Setup**

### **Local Development**
```bash
# Change code or train new model
# Model files auto-tracked with Git LFS

git add .
git commit -m "update: improve model accuracy"
git push
```

### **Deploy Changes**
```bash
# Vercel automatically deploys on git push
# Or manual:
vercel --prod

# Check health
curl https://your-project.vercel.app/health
```

### **Update Model**
```bash
# Replace model_artifacts/ files locally
# Git LFS automatically tracks changes

git add model_artifacts/
git commit -m "feat: update model with better accuracy"
git push

# Vercel redeploys automatically
```

---

## **Files Changed for Git LFS Deployment**

| File | Change | Reason |
|------|--------|--------|
| `.gitignore` | Remove `model_artifacts/` | LFS tracks it |
| `.env.example` | Remove model URLs | No cloud download |
| `vercel.json` | Remove MODEL env vars | Not needed |
| `api/model_utils.py` | Simplified load function | No cloud download |
| `.gitattributes` | Created by `git lfs track` | Tracks LFS files |

---

## **Complete Checklist**

- [ ] Git LFS installed (`git lfs version`)
- [ ] In project directory: `git lfs install`
- [ ] Tracked files: `git lfs track "model_artifacts/*"`
- [ ] `.gitattributes` committed
- [ ] `.gitignore` updated (model_artifacts NOT ignored)
- [ ] Model files added: `git add model_artifacts/`
- [ ] All committed and pushed: `git push`
- [ ] GitHub shows model files (not pointers)
- [ ] Deployed to Vercel: `vercel --prod`
- [ ] `/health` endpoint returns all files = true
- [ ] `/api/predict` works correctly

---

## **Quick Reference**

```bash
# Install LFS (one-time)
git lfs install

# Track files
git lfs track "model_artifacts/*"

# Add & commit
git add model_artifacts/ .gitattributes
git commit -m "feat: add model artifacts with Git LFS"
git push

# Deploy
vercel --prod

# Verify
curl https://your-project.vercel.app/health
```

---

## **Cost Summary**

| Resource | Cost |
|----------|------|
| GitHub LFS (free tier) | 1GB/month free |
| Vercel (free tier) | Unlimited deployments |
| **Total** | **FREE** ✅ |

---

**You're all set!** 🎉 Start with **Step 1** above.
