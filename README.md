# 🍅 TomatoScan — Tomato Disease Classifier


<div align="center">
  
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-1.8-orange?style=flat-square&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Accuracy-95.0%25-brightgreen?style=flat-square)
![ROC AUC](https://img.shields.io/badge/ROC--AUC-0.990-cyan?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-success?style=flat-square)


  
**AI-powered tomato leaf disease detection using XGBoost & SVM**

[🌐 Live Demo](https://noga-66.github.io/Tomato-Disease-Classifier/) 

![TomatoScan App Screenshot](screenshot.png)

</div>

---

## 📌 Overview

TomatoScan is a machine learning pipeline that classifies tomato leaf diseases from images. It detects two fungal pathogens:

| Disease | Pathogen | Symptoms |
|---|---|---|
| 🟠 **Early Blight** | *Alternaria solani* | Concentric-ring brown lesions on older leaves |
| 🔵 **Septoria Leaf Spot** | *Septoria lycopersici* | Small circular spots with dark borders |

The project includes:
- **Two trained models** — XGBoost (v2) and SVM RBF (v3)
- **Jupyter notebooks** with full pipeline
- **Interactive web app** with animated UI (deployable to GitHub Pages)
- **Results history** saved in browser localStorage

---

## 🏆 Results

### Model Comparison

| Model | Test Accuracy | ROC-AUC | F1-Score | 10-Fold CV |
|---|---|---|---|---|
| XGBoost v2 | 89.7% | 0.976 | 0.897 | 92.4% ± 7.8% |
| **SVM RBF v3** ⭐ | **95.0%** | **0.990** | **0.950** | **95.0% ± 5.0%** |
| RF+ET+SVM Ensemble | 90.0% | 0.993 | 0.899 | — |

### SVM v3 — Confusion Matrix (40 test samples)

```
                  Pred: EB    Pred: SLS
True: EB              18            2
True: SLS              0           20
```

**38 / 40 correct** — 2 false negatives only

### Per-Class Metrics (SVM v3)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Early Blight | 100% | 90% | 95% |
| Septoria Leaf Spot | 91% | 100% | 95% |

---

## 🧠 Model Architecture

```
Image (any size)
      │
      ▼
┌─────────────────────────────────────────────┐
│  Feature Extraction  (1884 dimensions)      │
│  ├── HOG (Histogram of Oriented Gradients)  │  1764 dims
│  ├── Color Histogram (3 × 32 bins RGB)      │    96 dims
│  └── Texture Statistics (quadrant stats)    │    29 dims
└─────────────────────────────────────────────┘
      │
      ▼
  StandardScaler  →  PCA (50–60 components, ~87% variance)
      │
      ▼
  SVM RBF Kernel  (C=10, gamma=scale, class_weight=balanced)
      │
      ▼
  Class Label + Probability Scores
```

---

## 📂 Project Structure

```
tomatoscan/
│
├── index.html                        # Interactive web app (GitHub Pages)
│
├── notebooks/
│   ├── Tomato_Disease_XGBoost_v2.ipynb   # XGBoost pipeline (v2)
│   └── Tomato_Disease_SVM_v3.ipynb       # SVM pipeline (v3) ← best model
│
├── tomato_data/
│   ├── early_blight/                 # 2 original images
│   └── septoria_leaf_spot/           # 58 original images
│
├── results/
│   ├── tomato_results_v2.json        # XGBoost metrics
│   └── tomato_results_v3.json        # SVM metrics
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/tomatoscan.git
cd tomatoscan
```

### 2. Install dependencies

```bash
pip install scikit-learn pillow numpy matplotlib seaborn
```

> ⚠️ No XGBoost package needed — `GradientBoostingClassifier` from scikit-learn implements the same algorithm.

### 3. Run the notebook

```bash
jupyter notebook notebooks/Tomato_Disease_SVM_v3.ipynb
```

Set `DATA_DIR` in Cell 2 to point to your dataset folder, then **Run All**.

### 4. Predict on a new image

```python
import pickle
from PIL import Image

# Load saved model
with open('tomato_svm_v3_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

# Predict
img = Image.open('your_leaf.jpg').convert('RGB')
label, proba = predict_image('your_leaf.jpg', bundle)
print(f"Prediction: {label}  |  Confidence: {max(proba)*100:.1f}%")
```

---

## 📊 Pipeline Details

### Data Augmentation (40+ transforms per image)

| Category | Transforms |
|---|---|
| Geometric | H-flip, V-flip, rotate ±15°/30°/45°/90°/180°/270° |
| Brightness | ×8 levels: 0.5 → 1.5 |
| Contrast | ×7 levels: 0.6 → 1.6 |
| Saturation | ×6 levels: 0.6 → 1.5 |
| Filter | Gaussian blur (r=1,1.5,2), Sharpen |
| Noise | ×8 random noise variants |
| Spatial | ×4 center crops |
| Combined | ×4 brightness+contrast combos |

**Result:** 2 Early Blight images → 100 samples per class (balanced)

### XGBoost Grid Search (v2)

Searched **27 combinations** (n_estimators × learning_rate × max_depth):

| Rank | n_estimators | lr | depth | CV Accuracy |
|---|---|---|---|---|
| ⭐ 1 | 300 | 0.08 | 4 | **95.17%** |
| 2 | 300 | 0.10 | 4 | 94.2% |
| 3 | 200 | 0.10 | 4 | 93.8% |

### Why SVM Beats XGBoost Here

| | SVM RBF ✅ | XGBoost ❌ |
|---|---|---|
| **Small dataset** | Maximum-margin classifier — optimal for small N, high D | Needs 500+ diverse real samples |
| **Class imbalance** | `class_weight='balanced'` in loss function | Requires careful sample weighting |
| **Feature space** | RBF kernel maps 1884 dims to separable space | 300 trees overfit augmented patterns |
| **Accuracy** | **95.0%** | 89.7% |

---

## 🌐 Web App

The `index.html` file is a fully self-contained interactive app with:

- 🎬 **Animated background** — floating particles + hex rings + data streaks
- 📤 **Image upload** with drag & drop
- ✕ **Remove & retry** button to swap images
- ⚡ **Demo buttons** for instant testing
- 📊 **Live results** — probability bars, confusion matrix, recommendation
- 💾 **Save to History** — persists in localStorage across sessions
- 📥 **Export JSON** — download prediction results
- 🗑️ **Delete / Clear All** history entries

### Deploy to GitHub Pages

1. Upload `index.html` to your repo root (rename to `index.html`)
2. Go to **Settings → Pages → Source: main / root**
3. Your app is live at `https://YOUR-USERNAME.github.io/tomatoscan/`

---

## 📓 Notebooks

| Notebook | Model | Accuracy | Cells |
|---|---|---|---|
| `Tomato_Disease_XGBoost_v2.ipynb` | GradientBoostingClassifier | 89.7% | 16 |
| `Tomato_Disease_SVM_v3.ipynb` | SVM RBF | **95.0%** | 16 |

Both notebooks include:
- Auto dataset resolution (supports ZIP archives)
- Data augmentation + loading
- Feature extraction
- PCA visualization
- Model training + grid search / model comparison
- 10-fold cross-validation
- Confusion matrix + ROC curve
- Feature importance
- Results dashboard
- Save model (.pkl) + results (.json)
- Predict on new image function

---

## 📦 Dependencies

```txt
scikit-learn>=1.3
numpy>=1.24
Pillow>=9.0
matplotlib>=3.7
seaborn>=0.12
```

---

## 📁 Dataset

The dataset used is a subset of the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease):

| Class | Original Images | After Augmentation |
|---|---|---|
| Early Blight | 2 | 100 |
| Septoria Leaf Spot | 58 | 100 |

> ⚠️ The current dataset is a small sample (part 1 of a split archive). For best results, use the full PlantVillage dataset with 1000+ images per class.

---

## 🔭 Future Improvements

- [ ] Add more disease classes (Late Blight, Leaf Mold, Mosaic Virus)
- [ ] Use full PlantVillage dataset (1000+ images/class)
- [ ] Try CNN / Transfer Learning (ResNet, EfficientNet)
- [ ] Real SMOTE via `imbalanced-learn` library
- [ ] REST API with Flask/FastAPI for production deployment

---

## 👩‍💻 Author

Built with  Nada Hossam ❤️ using Python, scikit-learn, and Claude AI.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
<sub>TomatoScan · XGBoost + SVM · HOG + Color + Texture features · PCA · 10-Fold CV</sub>
</div>

