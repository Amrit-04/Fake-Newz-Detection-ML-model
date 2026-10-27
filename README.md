# Fake News Detection & NLP Classification Pipeline 📰🔍

### Machine Learning Benchmark for Misinformation Detection using TF-IDF and Supervised Classifiers

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20Vectorization-blueviolet?style=flat)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project implements a complete Natural Language Processing (NLP) pipeline to detect and classify fake news articles. It benchmarks multiple supervised machine learning architectures against term-frequency inverse document frequency (TF-IDF) feature representations.

---

## 📊 Pipeline Architecture

```mermaid
flowchart LR
    Data[Raw News Dataset] --> Clean[Data Cleaning & Missing Value Removal]
    Clean --> Split[80/20 Stratified Train-Test Split]
    Split --> TFIDF[TF-IDF Feature Vectorizer]
    
    TFIDF --> NB[Multinomial Naive Bayes]
    TFIDF --> KNN[K-Nearest Neighbors k=5]
    TFIDF --> RF[Random Forest Ensemble]
    TFIDF --> SVM[Linear Support Vector Classifier]

    NB --> Eval[Classification Reports: Precision, Recall, F1]
    KNN --> Eval
    RF --> Eval
    SVM --> Eval
```

---

## 🔬 Methodology & Model Comparison

1. **Data Preprocessing:**
   - Cleans unstructured text and eliminates empty or corrupt records.
   - Normalizes text corpus to prevent sparse representation distortion.

2. **Feature Extraction (TF-IDF):**
   - Transforms n-gram text distributions into weighted numerical vectors:
     $$	ext{TF-IDF}(t, d, D) = 	ext{TF}(t, d) 	imes \log\left(rac{|D|}{1 + |\{d \in D : t \in d\}|}ight)$$
   - Penalizes common stopwords while amplifying highly discriminative topical vocabulary.

3. **Benchmarked Classifiers:**
   - **Multinomial Naive Bayes (`MultinomialNB`):** Highly effective probabilistic baseline for text classification assuming conditional word independence.
   - **K-Nearest Neighbors (`KNeighborsClassifier`):** Instance-based non-parametric classifier evaluating distance metrics in vector space.
   - **Random Forest (`RandomForestClassifier`):** Ensemble decision tree architecture robust against feature correlation and overfitting.
   - **Linear Support Vector Classifier (`SVC(kernel='linear')`):** Maximum-margin linear hyperplane separator optimizing decision boundary in high-dimensional sparse feature spaces.

4. **Evaluation Metrics:**
   - Evaluated using Scikit-Learn `classification_report`:
     - **Precision:** Exactness of positive fake news flags (reducing false alarms).
     - **Recall:** Sensitivity to catching deceptive misinformation.
     - **F1-Score:** Harmonic balance between precision and recall.

---

## 🚀 Running the Notebook

### Prerequisites

Ensure Python 3.9+ is installed along with the required scientific libraries:

```bash
pip install pandas numpy scikit-learn jupyter
```

### Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Amrit-04/Fake-Newz-Detection-ML-model.git
   cd Fake-Newz-Detection-ML-model
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook "ML Project.ipynb"
   ```

3. **Run all cells** to reproduce data preprocessing, model fitting, and evaluation reports.

---

## 📁 Repository Structure

```
Fake-Newz-Detection-ML-model/
├── ML Project.ipynb     # Complete notebook with data prep, TF-IDF, and 4 models
└── README.md            # Project documentation and benchmark explanation
```

---

## 📄 License

This project is licensed under the MIT License.