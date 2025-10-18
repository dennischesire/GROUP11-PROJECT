# TWEET SENTIMENT CLASSIFER
## Business Understanding
Social media platforms like Twitter heavily influence how people perceive brands, with thousands of users sharing opinions daily. Companies such as Apple and Google rely on understanding these sentiments to improve marketing, product design, and customer service. Manually analyzing this vast data is impractical, creating a need for an automated system that can classify tweets as positive, negative, or neutral. This project aims to build an NLP model to do exactly that by preprocessing tweet data, identifying sentiment patterns, training and evaluating classification models, and providing insights to help brands understand customer feedback. Success will be measured by achieving strong classification performance across all sentiment classes, with a target accuracy of around 80%.
## Data Understanding
The dataset to be used is from [data.world](https://data.world/crowdflower/brands-and-product-emotions). It's on judging emotions about brands and products specifically Apple and Google companies. 
**Columns**<br>
- `tweet_text` - Tweets by consumers of each brand and product.
- `emotion_in_tweet_is_directed_at` - The brand/product that the tweet is referring to.
- `is_there_an_emotion_directed_at_a_brand_or_product` - Emotion review. Positive, negative, I can't tell, among others.

Some more info on this dataset:
- Has 9093 rows and 3 columns.
- Has several missing values/data.
- Has several duplicates.
- Has no placeholders.

Some analysis was also done:
![Analysis](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/Screenshot%20(106).png)

![Analysis2](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/Screenshot%20(108).png)
## Data Preparation
The project involves data cleaning and text preprocessing to prepare tweets for NLP modeling. Data cleaning removes noise such as punctuation, numbers, URLs, and special characters, while stop words like "the" or "and" are removed to focus on meaningful words. Lemmatization is applied to reduce words to their base forms, helping the model recognize patterns across variations. These preprocessing steps ensure the text is clean, consistent, and ready for the sentiment analysis model.
## Modelling
For this sentiment analysis task, I’ve decided to work with four different models. Logistic Regression serves as a simple baseline, checking if the words in a tweet point more toward positive, negative, or neutral sentiment and giving a quick comparison point. Random Forest uses many small “decision trees” to predict sentiment based on word patterns and frequencies, making it robust against noisy or inconsistent text. XGBoost is a more advanced, tree-based model that learns from its past mistakes and often achieves high accuracy in text classification. Finally, Support Vector Machine (SVM) finds the best boundary to separate tweets with different sentiments and is particularly effective for high-dimensional text data.

To make the workflow smoother, I’m using pipelines that handle vectorization automatically, with both TF-IDF and Count Vectorizers. This setup allows me to preprocess the tweets and train the models consistently, and it makes it easy to compare how each vectorizer affects performance as I tune the models further.
## Evaluation
I tested Logistic Regression, Random Forest, XGBoost, and SVM using both TF-IDF and Count Vectorizers. Logistic Regression and the tree-based models often struggled with less frequent sentiment classes, while SVM showed more balanced performance across all classes. Using TF-IDF, SVM achieved 68% validation and 67% test accuracy, handling class imbalance effectively. Overall, SVM was the most reliable model for predicting tweet sentiments in this project.

Confusion Matrix:
![Analysis](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/Screenshot%20(109).png)
