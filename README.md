
# 🍓 Strawberry Leaf AI Disease Detection

> Final Year Project — FAST-NUCES Lahore | BS Computer Science | 2026

An AI-powered web application for real-time detection and classification of strawberry leaf diseases using deep learning and Explainable AI (XAI).

---

## 👥 Team

| Name | Roll No | Degree |
|------|---------|--------|
| Minahil Irfan | 22L-7821 | BS(CS) |
| Lyba Yaqub | 22L-7765 | BS(CS) |
| Rana Ahsan Ahmad | 22L-7895 | BS(SE) |

**Supervisor:** Ms. Anosha Khan

---

## 📌 About the Project

Strawberry leaf diseases like Leaf Scorch, Leaf Spot, and Powdery Mildew can devastate crop yields if not detected early. Traditional manual inspection is slow, error-prone, and inaccessible to many farmers.

This system solves that by:
- Automatically classifying strawberry leaf diseases from uploaded images
- Using **Grad-CAM** to visually highlight the affected regions (Explainable AI)
- Providing treatment recommendations for each detected disease
- Delivering results through a simple, farmer-friendly web interface

---

## 🎯 Key Results

| Model | Validation Accuracy | Weighted F1-Score |
|-------|-------------------|-------------------|
| DenseNet121 (fine-tuned) | 94.49% | 0.9491 |
| EfficientNetB0 (fine-tuned) | 97.68% | 0.9799 |
| ResNet50 (fine-tuned) | 96.50% | 0.9555 |
| **CNN Ensemble (Soft Voting)** | **98.84%** | **0.9883** |
| CNN Ensemble (Weighted Voting) | 98.84% | 0.9883 |

✅ **Deployed model: CNN Ensemble (Soft Voting)**

---

## 🌿 Disease Classes

The system detects **7 strawberry leaf disease classes** including:
- Angular Leaf Spot
- Anthracnose
- Leaf Scorch
- Leaf Spot
- Powdery Mildew
- Healthy

---

## 🛠️ Tech Stack

### Frontend
- React.js
- HTML, CSS, JavaScript

### Backend
- Python
- Django / Flask / FastAPI

### AI/ML
- TensorFlow / Keras
- PyTorch
- DenseNet121, EfficientNetB0, ResNet50
- Grad-CAM (Explainable AI)
- Albumentations (data augmentation)

### Database
- SQL / MongoDB

---

## 📁 Project Structure

```
strawberry-leaf-ai/
│
├── frontend/                  # React.js frontend
│   ├── public/
│   └── src/
│       ├── components/
│       ├── pages/
│       └── App.js
│
├── backend/                   # Django/Flask backend
│   ├── api/
│   ├── models/
│   └── utils/
│
├── ml/                        # Machine learning module
│   ├── datasets/
│   ├── models/
│   │   ├── densenet121/
│   │   ├── efficientnetb0/
│   │   └── resnet50/
│   ├── ensemble/
│   ├── gradcam/
│   ├── augmentation/
│   └── training/
│
├── database/                  # DB schemas and migrations
│
├── docs/                      # Report and documentation
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/strawberry-leaf-ai.git
cd strawberry-leaf-ai
```

### 2. Backend setup
```bash
cd backend
pip install -r requirements.txt
python manage.py runserver
```

### 3. Frontend setup
```bash
cd frontend
npm install
npm start
```

### 4. ML training (optional)
```bash
cd ml
pip install -r requirements.txt
python training/train.py
```

---

## 🔍 How It Works

1. User registers/logs in to the web app
2. User uploads a strawberry leaf image (JPG/PNG)
3. Backend preprocesses the image (resize to 224×224, normalize)
4. CNN Ensemble model classifies the disease
5. Grad-CAM generates a heatmap highlighting affected regions
6. Results displayed with: disease label, confidence score, XAI heatmap, and treatment advice

---

## 🌍 SDG Alignment

| SDG | Goal | How this project contributes |
|-----|------|------------------------------|
| SDG 2 | Zero Hunger | Early disease detection protects crop yields |
| SDG 9 | Industry, Innovation & Infrastructure | Brings AI into agriculture |
| SDG 12 | Responsible Consumption & Production | Reduces crop waste through timely intervention |

---

## 📊 Datasets Used

- [Kaggle Strawberry Leaf Dataset](https://www.kaggle.com) — ~2,500 images, 7 classes
- [Roboflow Strawberry Dataset](https://roboflow.com) — ~250 images, 4 classes
- [PlantVillage Dataset](https://plantvillage.psu.edu) — additional class balancing

Data augmentation applied: random flips, noise addition, brightness adjustment, blurring.

---

## 📄 License

This project is developed for academic purposes at FAST-NUCES Lahore.

---

## 🙏 Acknowledgements

- Supervisor: Ms. Anosha Khan
- FAST-NUCES Lahore
- PlantVillage, Kaggle, Roboflow for datasets
