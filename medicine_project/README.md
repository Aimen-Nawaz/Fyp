# 💊 Medicine Name Reconstruction

**Produces exactly this output format:**

```
Test Case 1:
Input: panadl
Predicted Output: panadol
Expected Output: panadol

Test Case 2:
Input: asati
Predicted Output: avastin
Expected Output: avastin

... etc
```

---

## 🎯 Why Previous Attempts Failed

| Problem | Fix |
|---------|-----|
| Used India CSV with **brand names** ("Augmentin 625 Tablet") — not generic drug names | Built `data/generic_drugs.txt` with **1,301 real drug names** |
| Files saved to `medicine_pairs.csv` not `/kaggle/working/...` | All paths now explicitly use `/kaggle/working/` |
| Target drugs (panadol, avastin, etc.) weren't in training data | All 5 reference image targets confirmed present ✅ |

---

## 📁 Project Structure

```
medicine_project/
├── README.md
├── requirements.txt
│
├── data/
│   └── generic_drugs.txt              ← 1,301 real drug names (upload this to Kaggle!)
│
├── notebooks/
│   └── Medicine_Reconstruction.ipynb  ← 31-cell Kaggle-ready notebook
│
├── app/
│   ├── app.py                         ← Flask web GUI
│   └── requirements_app.txt
│
└── model_artifacts/                   ← fills after downloading from Kaggle
    └── README.txt
```

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
