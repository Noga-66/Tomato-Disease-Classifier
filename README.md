🍅 TomatoScan — Tomato Disease Classifier
<div align="center">
Show Image
Show Image
Show Image
Show Image
Show Image
Show Image
AI-powered tomato leaf disease detection using XGBoost & SVM
🌐 Live Demo • 📓 Notebooks • 📊 Results • 🚀 Quick Start
</div>

📌 Overview
TomatoScan is a machine learning pipeline that classifies tomato leaf diseases from images. It detects two fungal pathogens:
DiseasePathogenSymptoms🟠 Early BlightAlternaria solaniConcentric-ring brown lesions on older leaves🔵 Septoria Leaf SpotSeptoria lycopersiciSmall circular spots with dark borders
The project includes:

Two trained models — XGBoost (v2) and SVM RBF (v3)
Jupyter notebooks with full pipeline
Interactive web app with animated UI (deployable to GitHub Pages)
Results history saved in browser localStorage


🏆 Results
Model Comparison
ModelTest AccuracyROC-AUCF1-Score10-Fold CVXGBoost v289.7%0.9760.89792.4% ± 7.8%SVM RBF v3 ⭐95.0%0.9900.95095.0% ± 5.0%RF+ET+SVM Ensemble90.0%0.9930.899—
SVM v3 — Confusion Matrix (40 test samples)
                  Pred: EB    Pred: SLS
True: EB              18            2
True: SLS              0           20
38 / 40 correct — 2 false negatives only
Per-Class Metrics (SVM v3)
ClassPrecisionRecallF1Early Blight100%90%95%Septoria Leaf Spot91%100%95%

🧠 Model Architecture
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

📂 Project Structure
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

🚀 Quick Start
1. Clone the repository
bashgit clone https://github.com/YOUR-USERNAME/tomatoscan.git
cd tomatoscan
2. Install dependencies
bashpip install scikit-learn pillow numpy matplotlib seaborn

⚠️ No XGBoost package needed — GradientBoostingClassifier from scikit-learn implements the same algorithm.

3. Run the notebook
bashjupyter notebook notebooks/Tomato_Disease_SVM_v3.ipynb
Set DATA_DIR in Cell 2 to point to your dataset folder, then Run All.
4. Predict on a new image
pythonimport pickle
from PIL import Image

# Load saved model
with open('tomato_svm_v3_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

# Predict
img = Image.open('your_leaf.jpg').convert('RGB')
label, proba = predict_image('your_leaf.jpg', bundle)
print(f"Prediction: {label}  |  Confidence: {max(proba)*100:.1f}%")

📊 Pipeline Details
Data Augmentation (40+ transforms per image)
CategoryTransformsGeometricH-flip, V-flip, rotate ±15°/30°/45°/90°/180°/270°Brightness×8 levels: 0.5 → 1.5Contrast×7 levels: 0.6 → 1.6Saturation×6 levels: 0.6 → 1.5FilterGaussian blur (r=1,1.5,2), SharpenNoise×8 random noise variantsSpatial×4 center cropsCombined×4 brightness+contrast combos
Result: 2 Early Blight images → 100 samples per class (balanced)
XGBoost Grid Search (v2)
Searched 27 combinations (n_estimators × learning_rate × max_depth):
Rankn_estimatorslrdepthCV Accuracy⭐ 13000.08495.17%23000.10494.2%32000.10493.8%
Why SVM Beats XGBoost Here
SVM RBF ✅XGBoost ❌Small datasetMaximum-margin classifier — optimal for small N, high DNeeds 500+ diverse real samplesClass imbalanceclass_weight='balanced' in loss functionRequires careful sample weightingFeature spaceRBF kernel maps 1884 dims to separable space300 trees overfit augmented patternsAccuracy95.0%89.7%

🌐 Web App
The index.html file is a fully self-contained interactive app with:

🎬 Animated background — floating particles + hex rings + data streaks
📤 Image upload with drag & drop
✕ Remove & retry button to swap images
⚡ Demo buttons for instant testing
📊 Live results — probability bars, confusion matrix, recommendation
💾 Save to History — persists in localStorage across sessions
📥 Export JSON — download prediction results
🗑️ Delete / Clear All history entries

Deploy to GitHub Pages

Upload index.html to your repo root (rename to index.html)
Go to Settings → Pages → Source: main / root
Your app is live at https://YOUR-USERNAME.github.io/tomatoscan/


📓 Notebooks
NotebookModelAccuracyCellsTomato_Disease_XGBoost_v2.ipynbGradientBoostingClassifier89.7%16Tomato_Disease_SVM_v3.ipynbSVM RBF95.0%16
Both notebooks include:

Auto dataset resolution (supports ZIP archives)
Data augmentation + loading
Feature extraction
PCA visualization
Model training + grid search / model comparison
10-fold cross-validation
Confusion matrix + ROC curve
Feature importance
Results dashboard
Save model (.pkl) + results (.json)
Predict on new image function


📦 Dependencies
txtscikit-learn>=1.3
numpy>=1.24
Pillow>=9.0
matplotlib>=3.7
seaborn>=0.12

📁 Dataset
The dataset used is a subset of the PlantVillage Dataset:
ClassOriginal ImagesAfter AugmentationEarly Blight2100Septoria Leaf Spot58100

⚠️ The current dataset is a small sample (part 1 of a split archive). For best results, use the full PlantVillage dataset with 1000+ images per class.


🔭 Future Improvements

 Add more disease classes (Late Blight, Leaf Mold, Mosaic Virus)
 Use full PlantVillage dataset (1000+ images/class)
 Try CNN / Transfer Learning (ResNet, EfficientNet)
 Real SMOTE via imbalanced-learn library
 REST API with Flask/FastAPI for production deployment


👩‍💻 Author
Built with ❤️ using Python, scikit-learn, and Claude AI.
