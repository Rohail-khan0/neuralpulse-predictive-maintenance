# 🏭 NeuralPulse: Industrial Predictive Maintenance System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

**NeuralPulse** is a cutting-edge **Machine Failure Prediction System** designed to minimize industrial downtime. Built using **Python (Streamlit)** and powered by **Machine Learning**, it predicts potential equipment failures in real-time based on sensor data (Temperature, RPM, Pressure, etc.).

The UI was conceptually designed using **Google Stitch** to ensure a professional, industrial-grade aesthetic.

---

## 📸 Project Screenshots

| **Login & Authentication** | **System Dashboard** |
|:--------------------------:|:--------------------:|
| ![Login Screen](./stitch/images/login_preview.png) | ![Dashboard](./stitch/images/dashboard_preview.png) |
| *Secure role-based access* | *Real-time health monitoring* |

| **Prediction Engine** | **Alert Logs** |
|:---------------------:|:--------------:|
| ![Prediction](./stitch/images/prediction_preview.png) | ![History](./stitch/images/history_preview.png) |
| *Live Failure Probability %* | *Historical audit trails* |

---

## ✨ Key Features

* **🧠 Advanced ML Integration:** Uses trained Logistic Regression/KNN models to predict failure probability with high accuracy.
* **🎨 Industrial UI Design:** Custom dark-mode interface with Glassmorphism effects, inspired by **Google Stitch** designs.
* **🔐 Secure Authentication:** Role-based login system for Operators and Managers (Supports Registration & Login).
* **📈 Real-Time Analytics:** Interactive line charts for sensor data (Temperature, Vibration, Torque).
* **⚠️ Smart Alerts:** Automatic flagging of critical failures with color-coded probability indicators (Red for Danger, Green for Safe).
* **📝 Audit History:** Exportable logs (CSV) of all machine predictions for maintenance review.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Custom CSS & Layouts)
* **Backend:** Python
* **Machine Learning:** Scikit-Learn (Logistic Regression, KNN), Pandas, NumPy
* **Design Tools:** Google Stitch (Beta), Google Antigravity (Cursor AI)
* **Data Handling:** Pickle (Model Serialization), JSON (User Auth)

---

## 🚀 Installation & Setup

Follow these steps to run the project locally:

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/NeuralPulse-Predictive-Maintenance.git](https://github.com/YourUsername/NeuralPulse-Predictive-Maintenance.git)
cd NeuralPulse-Predictive-Maintenance

📂 Project Structure
NeuralPulse/
├── app.py                  # Main Streamlit Application
├── best_model.pkl          # Trained ML Model
├── scaler.pkl              # Data Normalization Scaler
├── requirements.txt        # Python Dependencies
├── users.json              # User Database (JSON based)
├── stitch/                 # UI Assets (Images, Icons, CSS)
│   ├── images/             # Project Screenshots & Logos
│   └── styles/             # Custom CSS
└── README.md               # Project Documentation
