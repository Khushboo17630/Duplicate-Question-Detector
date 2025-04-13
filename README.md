# 🔍 Duplicate Question Detection

A Machine Learning project to detect duplicate questions using **Quora Question Pairs Dataset**. The model uses **advanced NLP techniques**, **feature engineering**, and is deployed via a simple **Flask web app**.

---

## 🚀 Project Workflow

### 📥 1. Data Collection
- **Dataset:** [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) (via Kaggle)
- **Columns Used:** `question1`, `question2`, `is_duplicate`

---

### 🧹 2. Text Preprocessing
- Lowercasing
- Removing HTML tags, special characters, stopwords
- Tokenization and stemming
- Punctuation cleanup

---

### 🧠 3. Feature Engineering
- **Token-Based Features:** Word match share, common word count
- **Length-Based Features:** Character/word count difference
- **Fuzzy Matching:** Ratio, partial ratio, token sort/set ratios
- **TF-IDF Vectorization:** Captures important tokens across questions

> ✅ Final feature matrix = [TF-IDF Vectors + Engineered Features]

---

### 📉 4. Dimensionality Reduction (Optional)
- **t-SNE** used for visualizing the data in lower dimensions  
*(Not used in the final model but helpful for exploration)*

---

### 🤖 5. Model Training
- **Algorithm:** Random Forest Classifier
- **Algorithm**: XGBoost Classifier (`XGBoost.XGBClassifier`)
- **Input:** Feature matrix from step 3
- **Output:** Binary label  
  - `1`: Duplicate  
  - `0`: Not Duplicate

---

### 🌐 6. Web App with Flask
- Users input two questions through a simple HTML form
- Backend:
  - Preprocesses text
  - Extracts features
  - Predicts duplication using trained model
- Output displayed on the same page with user-friendly message

---

## 💻 Tech Stack

| Component         | Libraries / Tools Used                                |
|------------------|--------------------------------------------------------|
| **Programming**   | Python                                                 |
| **Data Handling** | Pandas, NumPy                                          |
| **NLP & Cleaning**| NLTK, FuzzyWuzzy, BeautifulSoup                        |
| **ML Model**      | Scikit-learn (Random Forest)                           |
| **Vectorization** | TF-IDF                                                 |
| **Visualization** | t-SNE (via sklearn)                                   |
| **Deployment**    | Flask                                                  |
| **Frontend**      | HTML/CSS                                               |

---

## 📸 Screenshots of the web app interface
<b>1. Screenshot before entering question</><br>
![Screenshot (1060)](https://github.com/user-attachments/assets/52d105d6-0363-484e-9987-8e69d41f6297)



<b> 2.After entering question</b><br>
![Screenshot (1064)](https://github.com/user-attachments/assets/20366c16-50db-430c-bd7f-773373714510)

![Screenshot (1061)](https://github.com/user-attachments/assets/c709a4e9-fd5a-4049-80eb-7a910c180ddc)







## 📁 Folder Structure

```bash
.
├── app.py                   # Flask app
├── model.pkl                # Trained model
├── vectorizer.pkl           # TF-IDF vectorizer
├── templates/
│   └── index.html           # Frontend HTML
├── static/                  # CSS / Images (if any)
├── utils.py                 # Preprocessing and feature engineering functions
└── README.md                # Project documentation


📬 Contact

If you have any suggestions or questions, feel free to reach out via LinkedIn or open an issue!
