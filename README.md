## Twitter Sentiment Analysis (Natural Language Processing)

## Project Summary

This project uses Natural Language Processing (NLP) techniques to classify tweets related to Apple and Google. It does this by accomplishing these tasks:

Predict whether a tweet refers to an Apple or Google product.
Predict whether a tweet expresses a Negative, Neutral, or Positive sentiment. Built with libraries such as scikit-learn and NLTK, the project includes a complete machine learning pipeline from data preprocessing to model deployment.


## Data Understanding
The dataset used is tweet_product_company.csv, which contains real-world tweet data mentioning Apple and Google products. Each entry includes the tweet text, the referenced product (e.g., iPhone, Google), the associated company (Apple or Google), and a sentiment label (positive, negative, or neutral). It supports both binary and multiclass sentiment prediction and offers real-world variability, including emoji usage, slang, and abbreviations. We will explore the dataset to understand the kind of information it contains, the different features and their data types, as well as checking for things like missing values or unusual patterns. This will help us get a clear picture of the data before moving on to cleaning, preprocessing and vectorization.

## Problem Statement
Apple and Google, two of the world's leading tech companies, heavily depend on public perception to maintain their market positions and customer trust. As consumers increasingly voice their opinions on platforms like Twitter, understanding these sentiments has become very paramount for brand management, product development, and customer engagement strategies. This has therefore led to the need to develop algorithms that can provide sentiments for various opinions from users.

## Business Objectives
The primary client for this NLP project is Apple and Google. By analyzing sentiments from tweets about their products and those of their competitors, these tech giants can gain authentic feedback that traditional methods might miss. This real-time access to customer sentiment will enable them to quickly identify trends, preferences, and potential issues, facilitating proactive engagement and timely adjustments to their strategies.

## Project Objectives:
The main objectives of this project are:

* To explore and analyze the tweet data to understand the various disributions across the dataset.
* To apply text cleaning, preprocessing and vectorization techniques to prepare the Twitter data for effective model training.
* To develop a binary sentiment classifier to distinguish between positive and negative tweets, as a baseline.
* To extend the model to a multiclass classifier, that includes the neutral class.
* To evaluate classifier performance using appropriate metrics such as F1-score, precision and recall, particularly for imbalanced classes.
* To provide actionable, data-driven insights and recommendations that will guide these tech companies in leveraging sentiment analysis for future product developments.tevelopments.

## Exploratory Data Analysis
We performed a systematic investigation of the dataset to extract insights, evaluate feature distributions, assess the relationship between the feature and target variables, and identify anomalies, outliers or data quality issues. This was helpful in choosing the right modelling techniques. We also inclued visualizations showing sentiments and distributions as shown below:
