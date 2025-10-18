<<<<<<< HEAD
# Twitter Sentiment Analysis
## Business Understanding
**Stakeholder & Problem Statement**: Sarah Chen, Senior Product Manager for iOS at Apple Inc., faces challenges in keeping up with the vast amount of customer feedback on Twitter. With thousands of daily mentions, manually analyzing sentiment is slow, inconsistent, and difficult to scale. This leads to delayed responses to product issues and missed complaints, ultimately affecting customer satisfaction and the quality of Apple products.

The main objective is to build a machine learning system that automatically classifies Twitter sentiment about Apple products to enable real-time customer feedback monitoring and faster issue resolution.
### Advanced ML summary
We developed an automated sentiment classifier to help Apple’s product team monitor customer feedback on Twitter, with a focus on detecting complaints. Using 9,093 human-labeled tweets, the system categorizes sentiment as Positive, Negative, or Neutral.

**Data & Preprocessing**: Tweets were cleaned (lowercasing, URL removal, tokenization, lemmatization, stopword removal) and vectorized using TF-IDF (5,000 features, (1-2) n-grams). Severe class imbalance (7% negative) was addressed with SMOTE to generate synthetic negative examples.

**Modeling & Evaluation**: Logistic Regression, Random Forest, and SMOTE-enhanced Logistic Regression were tested. SMOTE Logistic Regression achieved the best balance of recall and precision. On the test set, the model reached 61.7% accuracy with 50% negative tweet recall, detecting 5.5× more complaints than previous approaches while maintaining balanced performance across classes.

Impact: The system automates complaint detection, enables faster response to negative feedback, and delivers actionable insights for product strategy.
## Data Understanding
The dataset is from [data.world](https://data.world/crowdflower/brands-and-product-emotions). It's on Twitter Sentiment Analysis (Apple vs. Google). Human raters collected the data by manually labelling the sentiments. A total of 9093 tweets with 3 columns.
**Column Overview**:
- tweet_text: The actual content of the tweet (our primary feature)
- emotion_in_tweet_is_directed_at: The brand/product being mentioned (Apple, Google, etc.)
- is_there_an_emotion_directed_at_a_brand_or_product: The sentiment label (Positive/Negative/Neutral emotion)

A bit of EDA(Exploratory Data Analysis) is done. 
There is some class imbalance as shown below:

<p align ='center'>Class imbalance in the sentiments</p>

![Analysis 1](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(117).png)

Text Characteristics by sentiment
![Analysis 2](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(110).png)

Brand and sentiment Cross-Analysis
![Analysis 3](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(111).png)

among others.
## Data Preparation
=======
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
>>>>>>> 624ceb7b45b37675bade63989c2a4d62fa2a5e08
