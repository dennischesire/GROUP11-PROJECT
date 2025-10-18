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
