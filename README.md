This repository contains hands-on implementations of core machine learning topics, divided into 9 parts:

📌 1. Data Preprocessing
Before feeding data into any machine learning model, it must be cleaned and prepared. This step includes:

Handling missing values – filling or removing null entries.

Encoding categorical variables – converting text labels to numbers (e.g., one-hot encoding).

Feature scaling – normalizing or standardizing features for better model performance.

Splitting the dataset – dividing data into training and testing sets.

📌 2. Regression
Regression models predict continuous numeric outcomes. Techniques include:

Simple Linear Regression – models the relationship between two variables.(Use to predict the salary for the new employee based on no.of years experince)

Multiple Linear Regression – includes multiple input variables.(use to predict to find in which startup have to invest based on the given features)

Polynomial Regression – fits a nonlinear curve to the data.(Salary Prediction)

Support Vector Regression (SVR) – uses SVM principles to predict continuous output.(Salary prediction)

Decision Tree & Random Forest Regression – tree-based methods for flexible, non-linear predictions.(Salary Prediction)

📌 3. Classification
Classification is used to predict categorical outcomes (like spam or not spam). Algorithms include:

Logistic Regression – predicts binary outcomes.(to predict which customer will buy new SUV)

K-Nearest Neighbors (KNN) – classifies based on nearest data points.(to predict which customer will buy new SUV)

Support Vector Machines (SVM) – finds the best margin to separate classes. (to predict which customer will buy new SUV)

Naive Bayes – a probabilistic model based on Bayes’ Theorem.(to predict which customer will buy new SUV)

Decision Tree & Random Forest – rule-based models for accurate classification.(to predict which customer will buy new SUV)

📌 4. Clustering
Unsupervised learning to group similar data points. Common algorithms:

K-Means Clustering – partitions data into K groups based on similarity.(metric measuring spending of each customer)

Hierarchical Clustering – builds a tree-like structure of clusters (dendrogram).(metric measuring spending of each customer)

📌 5. Association Rule Learning
Used to find relationships or patterns in large datasets, such as market basket analysis.

Apriori Algorithm – finds frequent itemsets using a level-wise search. (if you buy this product you will get that product for free i.e., we will measure the chance if one buy x product what will be the chances of buying y product)

Eclat Algorithm – uses a depth-first approach for better performance in dense datasets.(if you buy this product you will get that product for free i.e., we will measure the chance if one buy x product what will be the chances of buying y product)

📌 6. Reinforcement Learning
Agents learn to take actions in an environment to maximize cumulative reward.

Upper Confidence Bound (UCB) – balances exploration and exploitation in multi-armed bandit problems.(best ad that will convert the maximum customers to click on the ad -- optimizing the click through rate of some ads)

Thompson Sampling – probabilistic method for choosing actions in uncertain environments.(best ad that will convert the maximum customers to click on the ad -- optimizing the click through rate of some ads)

📌 7. Natural Language Processing (NLP)
Machines interpret and understand human language. This part covers:

Text Preprocessing – cleaning text (removing punctuation, stopwords, etc.).

Bag of Words Model – converts text into numerical vectors.

Naive Bayes Classifier – used for sentiment analysis or spam detection.

(Sentiment analysis-i.e., whether the review is positive or negative)

📌 8. Deep Learning
A subfield of ML using neural networks for complex tasks.

Artificial Neural Networks (ANN) – models inspired by the human brain for tabular data.(Predicting Customer Churn with TensorFlow)

Convolutional Neural Networks (CNN) – specialized for image processing and computer vision tasks.(Image CLassification)

📌 9. Dimensionality Reduction
Reduces the number of input features while preserving data structure.

Principal Component Analysis (PCA) – transforms features into uncorrelated components. (Customer Segment-Recommendation system)

Linear Discriminant Analysis (LDA) – maximizes class separability.(Customer Segment-Recommendation system)

Kernel PCA – non-linear form of PCA using kernel tricks.(Customer Segment-Recommendation system)
