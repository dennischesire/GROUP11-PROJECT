# Twitter Sentiment Analysis
## Business Understanding
**Stakeholder & Problem Statement**: Sarah Chen, Senior Product Manager for iOS at Apple Inc., faces challenges in keeping up with the vast amount of customer feedback on Twitter. With thousands of daily mentions, manually analyzing sentiment is slow, inconsistent, and difficult to scale. This leads to delayed responses to product issues and missed complaints, ultimately affecting customer satisfaction and the quality of Apple products.

The main objective is to build a machine learning system that automatically classifies Twitter sentiment about Apple products to enable real-time customer feedback monitoring and faster issue resolution.
### Advanced ML summary
We developed an automated sentiment classifier to help Apple’s product team monitor customer feedback on Twitter, with a focus on detecting complaints. Using 9,093 human-labeled tweets, the system categorizes sentiment as Positive, Negative, or Neutral.

**Data & Preprocessing**: Tweets were cleaned (lowercasing, URL removal, tokenization, lemmatization, stopword removal) and vectorized using TF-IDF (5,000 features, (1-2) n-grams). Severe class imbalance (7% negative) was addressed with SMOTE to generate synthetic negative examples.

**Modeling & Evaluation**: Logistic Regression, Random Forest, and SMOTE-enhanced Logistic Regression were tested. SMOTE Logistic Regression achieved the best balance of recall and precision. On the test set, the model reached 61.7% accuracy with 50% negative tweet recall, detecting 5.5× more complaints than previous approaches while maintaining balanced performance across classes.

**Deployment**: Our analysis has been successfully deployed as a production-ready web application, enabling product teams to leverage our sentiment classification model in their daily workflow. This addition transforms the notebook from static analysis to a dynamic, implemented solution that delivers immediate business value

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

Among others.
## Data Preparation
- Cleaning & Processing: Removed duplicates, empty, and very short tweets; standardized sentiment labels to Positive, Negative, and Neutral.

<p align = 'center'>Sentiment Distribution after Mapping</p>

  ![Analysis 1](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(112).png)
- Dataset Creation: Built an Apple-focused subset of 5,633 tweets while retaining the original dataset.
- Text Preprocessing: Applied cleaning, tokenization, and lemmatization.
- Dataset Overview: 5,633 tweets with 54% Neutral, 38% Positive, 8% Negative.

<p align = 'center'> Cleaned tweet Analysis</p>

![Analysis 3](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(120).png)
Key Consideration: Addressed class imbalance during modeling; proceeded with three-class classification for nuanced sentiment insights.

The dataset is then split into a training set, a validation set, testing set.
## Modelling
Two major models were evaluated for sentiment classification: Logistic Regression (baseline) and Random Forest.

**The dataset used for the results in modelling is the validation set, not the testing set as it is used in the evaluation step.**

### 1. Logistic Regression
The baseline Logistic Regression model achieved an overall accuracy of 63.4%. It performed well on neutral tweets (precision = 0.65, recall = 0.79) but struggled with negative sentiment detection (recall = 0.09). The model showed moderate performance for positive tweets (f1 = 0.55). Overall, it indicated a bias toward neutral sentiment, revealing the need for better class balance handling.

<p align = 'center'> Logistic Regression(Baseline model)</p>

![Analysis 5](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(113).png)

### 2. Initial Comparison (No Tuning)
Logistic Regression outperformed Random Forest overall.
Accuracy: 0.634 vs 0.608

Negative recall was very low for both (0.09 vs 0.078), indicating that rare negative complaints were poorly captured.
F1-scores showed Logistic Regression performed better on Positive and Negative classes, while Random Forest slightly better on Neutral.

<p align = 'center'> Logistic Regression Vs Random Forest</p>

![Analysis 6](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(114).png)

### 3. Logistic Regression with Class Balancing
Balancing class weights significantly improved the model’s ability to detect negative tweets.

Negative recall jumped from 0.09 to 0.50 (+0.41), while overall accuracy slightly decreased (0.634 to 0.582).
This trade-off was favorable since detecting negative feedback was the primary objective. 
.
<p align = 'center'> Logistic Regression with classes balanced</p>

![Analysis 7](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(115).png)

### 4. SMOTE with Logistic Regression
Using SMOTE to generate synthetic negative examples further improved Negative F1 (0.340 to 0.360) and Precision (0.260 to 0.290) with minimal impact on accuracy (0.582 to 0.590).

Negative recall remained high (0.484), close to the auto-balanced model.

<p align = 'center'>SMOTE with Logistic Regression</p>

![Analysis 8](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(118).png)

### 5. Logistic Regression Model Comparison
We evaluated three Logistic Regression variants for detecting negative tweets. The original model (no weighting) had the highest overall accuracy (0.634) but very low negative recall (0.094), missing most complaints. Applying class weights (Auto-Balanced) dramatically improved negative recall to 0.500, though precision dropped, resulting in more false positives. Using SMOTE to generate synthetic negative examples slightly reduced recall (0.484) but increased precision (0.287) and achieved the highest Negative F1 (0.360), providing the best balance between detecting complaints and minimizing false alerts. Overall, SMOTE Logistic Regression was the most effective model for actionable complaint detection.

<p align = 'center'>Final Comparison Between the Logistic models</p>

![Analysis 9](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(119).png)

## Evaluation
The testing set is used here. 

The SMOTE Logistic Regression model, identified as the best-performing model, was evaluated on the test set and achieved an overall accuracy of 61.7%. Performance on individual sentiment classes demonstrates its effectiveness in capturing customer feedback:
- Negative Tweets: Recall 50.0%, Precision 32.1%, F1-Score 39.1% (86 tweets)
- Neutral Tweets: Recall 64.0%, Precision 70.0%, F1-Score 67.0% (611 tweets)
- Positive Tweets: Recall 60.0%, Precision 60.0%, F1-Score 60.0% (430 tweets)

Compared to the validation set, the test set results show slight improvements across all key metrics, with accuracy increasing from 59.1% to 61.7%, negative recall from 48.4% to 50.0%, negative precision from 28.7% to 32.1%, and negative F1 from 36.0% to 39.1%.

These results indicate that the model generalizes well to unseen data. It is particularly effective at detecting negative tweets, which is the primary business objective, while maintaining balanced performance for neutral and positive sentiments. Overall, the model provides a reliable automated solution for monitoring customer complaints and delivering actionable insights for product management.

<p align = 'center'>Evaluation Using the testing set</p>

![Analysis 10](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(116).png)

## Conclusion
We developed and validated a production-ready sentiment analysis system that automatically detects customer complaints for Apple. The SMOTE Logistic Regression model achieves 50% negative tweet detection, generalizes well to unseen data, and balances recall with overall performance. By handling class imbalance effectively, it transforms Twitter data into actionable insights, giving the product team real-time visibility into customer sentiment and emerging issues.

## Recommendations
- Deploy SMOTE Logistic Regression to cloud environment
- Set up real-time Twitter API integration
- Create basic alerting system for product team
- Establish model performance monitoring

## Deployment
We deployed the SMOTE Logistic Regression model on Streamlit, creating an easy-to-use tool for individuals to analyze customer tweets in real time. It helps them quickly detect complaints, understand customer sentiment, and respond to issues more efficiently.

Below is the link to the deployed model on Streamlit:

[Apple Sentiment Analyzer](https://dennischesire-group11-project-app-rjjfiy.streamlit.app/)
 <p align = 'center'> User Interface </p>

 ![Analysis](https://github.com/dennischesire/GROUP11-PROJECT/blob/ivy/images/Screenshot%20(121).png)
