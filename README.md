# Truth or Deception: A Scalable Approach to Fake News Detection with Ethical AI

This project explores the detection of fake news using **machine learning models based solely on textual content** (titles and article body). We use PySpark pipelines and multiple classification algorithms to build scalable and explainable models for misinformation detection.

---

## Project Structure

Project/  
├── data/  
&emsp;├──Fake.csv  
&emsp;├──True.csv  
&emsp;└──WELFake_Dataset.csv  
├── Notebook 1 - Fake News Datasets Preprocessing and Exploratory Analysis.ipynb # Cleaning, lemmatization, TF-IDF vectorization  
├── Notebook 2 - Fake News Classification Models.ipynb # Training and evaluating classification models  
├── Notebook 3 - Sentiment Analysis.ipynb # Extracting sentiment-based features using TextBlob  
└── README.md # Project overview and execution guide

## Requirements

This project requires the following libraries:

- Python ≥ 3.9  
- Apache Spark (pyspark)  
- nltk 
- textblob 
- scikit-learn 
- matplotlib, seaborn 
- langid 
- regex

To install the main dependencies:

--> pip install pyspark nltk textblob scikit-learn langid matplotlib seaborn tqdm joblib click regex


## Project description

Our project aims to develop and compare multiple machine learning models to accurately detect fake news. By analyzing labeled news data, we assess how well different classifiers, such as Logistic Regression, Naive Bayes, Decision Tree, and Random Forest perform at identifying fake news articles.


## Execute the notebooks in the following order:

1. **Notebook 1 – Data Preprocessing**
    - Merges and shuffles real/fake news from ISOT
    - Cleans text: lowercasing, punctuation/stopword removal
    - Applies lemmatization
    - Converts text into TF-IDF features for ML models


2. **Notebook 2 – Model Training & Evaluation**
    - Loads preprocessed data
    - Trains ML models: Logistic Regression, Naïve Bayes, Decision Tree, Random Forest
    - Evaluates performance: recall, precision, accuracy, F1
    
    
3. **Notebook 3 – Sentiment Analysis**
    - Uses TextBlob to extract polarity and subjectivity features
    - Enriches dataset with emotional tone as an additional feature
    - Re-runs model evaluation using sentiment-enhanced inputs


## Outputs

- Cleaned and preprocessed dataset
- Evaluation results for all models on ISOT and WELFake test sets
- Confusion matrix and performance metrics per model
- Sentiment Analysis using TextBlob

## Authors

Sofia Karanukhova  
Tim Gotschim  
Felix Gehmair  
Marcel Tobler

## License

Copyright (c) 2025 
Team FNF – Fight Fake News (Sofia Karanukhova, Tim Gotschim, Felix Gehmair, Marcel Tobler)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to use, copy, modify, merge, publish, and distribute the Software for **educational and non-commercial purposes**, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
2. You may not use the Software for commercial or surveillance purposes without prior written permission.
3. The Software is provided "as is", without warranty of any kind.


### GDPR and Ethical Use Addendum

This software and all associated datasets are to be used **in full compliance with the European General Data Protection Regulation (GDPR)**:

- Do not use the software with datasets containing personally identifiable information (PII), unless such data is properly anonymized.
- Any processing of personal data must ensure that individuals are no longer identifiable (Recital 26 GDPR).
- This software must not be used to build or deploy systems that could harm individuals through misinformation, profiling, or discrimination.
- When using third-party datasets, you must respect the original licenses and verify that the data collection complied with privacy laws.

This project was developed for academic purposes only and reflects the ethical standards of transparent, responsible, and fair data science.

