# 🚀 Quick Start - Deploy to Vercel in 5 Minutes

## **1️⃣ Setup Git LFS** (5 min)

Store model files directly in your GitHub repo using Git LFS.

```bash
# Install Git LFS (one-time)
choco install git-lfs          # Windows
# or brew install git-lfs       # macOS
# or sudo apt install git-lfs   # Linux

# In your project directory
cd medicine_project
git lfs install
git lfs track "model_artifacts/*"
git add model_artifacts/ .gitattributes
git commit -m "feat: add model artifacts with Git LFS"
git push
```

---

## **2️⃣ Deploy**

```bash
# Install Vercel CLI (one-time)
npm install -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

---

## **3️⃣ Test**

```bash
# Check health
curl https://your-project.vercel.app/health

# Test prediction
curl -X POST https://your-project.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"input":"panadl"}'
```

---

## ✅ **Done!** Your app is live.

**Visit:** https://your-project.vercel.app

---

## **Why Git LFS?**

✅ No cloud storage needed (Google Drive, S3, etc.)
✅ Model files in your repo
✅ Automatic Vercel deployment
✅ Free (1GB/month GitHub LFS)

**Full guide:** [GIT_LFS_DEPLOYMENT.md](GIT_LFS_DEPLOYMENT.md)

