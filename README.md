# 🩺 SwasthCare 24/7

**SwasthCare 24/7** is an AI-based alternative medicine recommendation system that suggests suitable drugs based on a given medicine name. It uses **Natural Language Processing (NLP)** and **machine learning** powered by **SentenceTransformer BERT (all-MiniLM-L6-v2)** to make healthcare more accessible, affordable, and intelligent.

---

## 🚀 Features

- 🔍 Search for medicines and get complete details  
- 💊 Suggests alternative medicines based on similarity  
- 🤖 NLP + SentenceTransformer (BERT) for smart recommendations  
- 📃 View composition, uses, side effects  
- 🌍 Supports both urban and rural users  
- 💬 Fast, responsive frontend using React + Vite  

---

## 🛠️ Tech Stack

- **Frontend:** React + Vite  
- **Backend:** Flask (Python)  
- **ML Model:** SentenceTransformer BERT (`all-MiniLM-L6-v2`)  
- **Dataset:** `Medicine_Details.csv`  
- **Optional OCR:** Tesseract.js for image-based medicine input  
- **Communication:** REST API with JSON  

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/swasthcare-ai.git
cd swasthcare-ai
 
then
cd backend
pip install -r requirements.txt
python app.py

to run front end 
cd frontend
bun install          # or npm install / yarn install
bun dev              # or npm run dev / yarn dev
