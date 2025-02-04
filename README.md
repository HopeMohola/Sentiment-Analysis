# Sentiment and Emotion Classification using Machine Learning

## Overview
This project focuses on sentiment and emotion classification from text data using several machine learning models. The goal is to train models on preprocessed text data to predict the sentiment and emotion of given sentences. 

## The following models are implemented:
Logistic Regression
Support Vector Machine (SVM)
Random Forest

## Installation
To run this project, ensure that you have the necessary libraries installed:
pip install pandas numpy scikit-learn matplotlib seaborn
For TensorFlow (if needed for neural networks or deeper models):
pip install tensorflow (But note that i didnt run neural networks for this project)

## Dataset
The dataset consists of textual data (sentences) labeled with their corresponding sentiment and emotion:
emotion_cleaned: Contains sentences and their corresponding emotion labels.
sentiment_cleaned: Contains sentences and their corresponding sentiment labels.
Columns:
sentence: The textual data (sentence).
emotion: The emotion label for emotion classification.
sentiment: The sentiment label for sentiment classification.

## Models
This project implements three machine learning models to classify sentiment and emotion:
1. Logistic Regression
A linear model commonly used for binary classification tasks. Logistic Regression is used to predict binary sentiment or emotion labels.

2. Support Vector Machine (SVM)
A model that works by finding the hyperplane that best separates the data into classes. It is particularly effective in high-dimensional spaces, which is why it works well with text classification.

3. Random Forest Classifier
An ensemble model based on decision trees. Random Forest uses multiple trees to make predictions, which helps reduce overfitting and increases accuracy.

## Workflow
Data Preprocessing
Text Cleaning: Raw text data is preprocessed to remove unwanted characters, punctuation, and stopwords.
Feature Extraction: We use TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features suitable for the models.
Data Splitting: The dataset is split into training and testing sets to evaluate the models' performance.

Model Training and Evaluation
Logistic Regression, SVM, and Random Forest models are trained on the processed data.
Performance Evaluation: Models are evaluated using several metrics, including:
Accuracy: The proportion of correct predictions.
Precision: The ability of the model to correctly predict positive samples.
Recall: The ability of the model to capture all positive samples.
F1-Score: A weighted average of Precision and Recall, providing a balance between the two.

## Running the Models
To run the models, simply load the preprocessed dataset and train each model. Each model will output its performance metrics, allowing for comparison to select the best performing model for sentiment or emotion classification.

## Key Evaluation Metrics
Logistic Regression
Sentiment Accuracy: 72.3%
Emotion Accuracy: 89.9%

Support Vector Machine (SVM)
Sentiment Accuracy: 72.0%
Emotion Accuracy: 72.0%

Random Forest
Sentiment Accuracy: 72.2%
Emotion Accuracy: 72.2%

## Conclusion
Logistic Regression outperforms the other models in emotion classification with an accuracy of 89.9%.
All models have similar performance for sentiment classification, with accuracies around 72%.
Logistic Regression is the best choice for emotion classification(Because its binary, negative or positive), while all models perform similarly for sentiment tasks.
