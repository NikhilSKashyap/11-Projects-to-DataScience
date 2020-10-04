# 11 PROJECTS TO DATA SCIENCE

In this 11 project series, we will explore Data Science concepts using different Kaggle datasets.

The main goal of this approach is to -
1. Understand Data Science concepts
2. Get comfortable handling data and building models

Largely there are three major categories of Machine Learning they are **Supervised**, **Unsupervised**, **Semi-supervised**. In this series, we will focus on **Supervised** and **Unsupervised**.
<br>
<br>

## SUPERVISED LEARNING

**Supervised learning** is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output.

The goal is to approximate the mapping function so well that when you have new input data (x) that you can predict the output variables (Y) for that data.

There are two types of Supervised Learning techniques: **Regression** and **Classification**. **Classification** separates the data, **Regression** fits the data.
<br>

### REGRESSION

**Regression** is a technique that aims to reproduce the output value. We can use it, for example, to predict the price of some product, like the price of a house in a specific city or the value of a stock. There is a huge number of things we can predict if we wish.

1. [Logistic Regression](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/1.LogisticRegression.ipynb)<br>
    In this very first project, we will use [Titanic dataset](https://www.kaggle.com/c/titanic/overview) to explore the basics of python programming for Data Science. You'll learn the following - 
    * Read data files using pandas. 
    * Perform operations on the data using pandas.
    * Visualize data using Matplotlib and Seaborn. 
    * Logistic Regression and how to build it using scikit-learn. 

2. [Linear Regression](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/2.LinearRegression.ipynb)<br>
    Linear regression is one of the very basic forms of machine learning where we train a model to predict the behavior of the data based on some input variables. We will use [House price prediction](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset to understand the concepts. In this lesson, you'll learn - 
    * In-depth understanding of handling missing data.
    * Extract useful features using the Correlation Matrix.
    * Handling outliers and skewed data.
    * Build a Linear Regression model using scikit-learn and analyze the performance using RMSE.

3. [Decision Trees](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/3.DT-RF-LGBM.ipynb)<br>
    Decision Tree is a Supervised learning technique that can be used for both Classification and Regression problems. The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data). In Decision Trees, for predicting a class label for a record we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. Based on the comparison, we follow the branch corresponding to that value and jump to the next node.<br>
    Decision Trees tend to overfit on their training data, making them perform badly if data previously shown to them doesn’t match to what they are shown later. They also suffer from high variance. Decision Trees can also create biased Trees if some classes dominate over others. The drawback of Decision Trees is that they look for a locally optimal and not a globally optimal at each step.<br>

    There are different strategies to overcome these drawbacks. Ensemble methods combine several decision trees classifiers to produce better predictive performance than a single decision tree classifier. The main principle behind the ensemble model is that a group of weak learners come together to form a strong learner, thus increasing the accuracy of the model.<br>

    The two most common types of Ensemble learning are **Bagging method** and **Boosting method**.<br>

    **Bagging** is a way to decrease the variance in the prediction by generating additional data for training from the dataset using combinations with repetitions to produce multi-sets of the original data.<br>

    **Boosting** is an iterative technique that adjusts the weight of an observation based on the last classification. If an observation was classified incorrectly, it tries to increase the weight of this observation.<br>

    In this lesson, we will use [NYC Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) dataset to explore Decision trees, the Bagging method(RandomForest) & the Boosting method(LightGBM). You will learn - 
    * Handle large data
    * How to calculate the distance between two points which isn't one the cartesian plane?
    * How to plot data on a map using Folium?
    * Build Linear Regression, Decision Tree, RandomForest, and LightGBM models.
    * Compare the performances of models using RMSE values. 
<br>

### CLASSIFICATION

Classification is a technique that aims to reproduce class assignments. It can predict the response value and the data is separated into “classes”. Examples? Recognition of a type of car in a photo is this mail spam or a message from a friend, or what the weather will be today.

4. [Support Vector Machines(SVM)](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/4.SupportVectorMachines.ipynb)<br>
    Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression, and outlier detection.

    The advantages of support vector machines are:
    * Effective in high dimensional spaces.
    * Still effective in cases where the number of dimensions is greater than the number of samples.
    * Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
    * Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

    The disadvantages of support vector machines include:
    * If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
    * SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.<br>
    (https://scikit-learn.org/stable/modules/svm.html)

    In this lesson using [Credit Card Fraud Prediction](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset, you'll learn - 
    * What is classification?
    * Undersampling
    * Build Linear Regression, SVC, and RandomForestClassifier models.
    * Compare them using the ROC-AUC score and F1-Score. 

5. [K-Nearest Neighbors(KNN)](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/05.KNearestNeighbors.ipynb)<br>
    K-Nearest Neighbors is one of the simplest supervised learning classification algorithms. The KNN algorithm assumes that similar things are near to each other. 

    Using [Pima Indians Diabetes](https://www.kaggle.com/uciml/pima-indians-diabetes-database) dataset, we will learn - 
    * How to standardize features by removing the mean and scaling to unit variance using StandardScaler from scikit learn?
    * KNN algorithm and how to build it.


6. [Bonus](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/6.Bonus.ipynb)<br>
    In this lesson, we will use [Telco Churn Prediction](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/6.telecom_churn_data.zip) dataset and build 12 different models on the same data. You will perform - 
    * In-depth EDA and Feature Engineering.
    * Handle Data Imbalance using SMOTE or ADASYN.
    * Build Baseline Models using Logistic Regression, RidgeClassifier, SGDClassifier, KNN, LinearSVC, SVC, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, ADABoostClassifier, GradientBoostingClassifier, and XGBoostClassifier. 
    * Compare the models using "roc_score" and "recall_score".
    * Select multiple models and tune their hyperparameters for better performance. 
    * Compare models using f1-score.
<br>

## UNSUPERVISED LEARNING

Unsupervised learning is where you only have input data (X) and no corresponding output variables.<br>
The goal for unsupervised learning is to model the underlying structure or distribution in the data to learn more about the data.<br>
These are called unsupervised learning because unlike supervised learning above there are no correct answers and there is no teacher. Algorithms are left to their devices to discover and present the interesting structure in the data.<br>
In unsupervised techniques, we have clustering, association, and dimensionality reduction.
<br>

### CLUSTERING

Clustering is used to find similarities and differences. It groups similar things. Here we don’t provide any labels, but the system can understand data itself and cluster it well. Unlike classification, the final output labels are not known beforehand.

This kind of algorithm can help us solve many obstacles, like create clusters of similar tweets based on their content, find groups of photos with similar cars, or identify different types of news.

7. [K-Means](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/7.KMeans.ipynb)<br>
    K-Means is one of the popular unsupervised clustering algorithms. Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.

    The algorithm works as follows:<br>
    * First we initialize k points, called means, randomly. 
    * We categorize each item to its closest mean and we update the mean’s coordinates, which are the averages of the items categorized in that mean so far.
    * We repeat the process for a given number of iterations and in the end, we have our clusters.

    Using [Mall Customer Segmentation](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python) dataset, we will learn the following - 
    * How to determine the number of clusters in the dataset using the Elbow method?
    * How to build an unsupervised clustering model using K-Means?
    * Interactive 3D visualization using Plotly.

8. [DBSCAN](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/8.DBSCAN.ipynb)<br>
    Density-based spatial clustering of applications with noise (DBSCAN) is a well-known data clustering algorithm that is commonly used in data mining and machine learning. DBSCAN identifies distinctive groups/clusters in the data, based on the idea that a cluster in data space is a contiguous region of high point density, separated from other such clusters by contiguous regions of low point density.

    We will use [Target store dataset](https://www.kaggle.com/ben1989/target-store-dataset) in this lesson. You will learn - 
    * Limitations of K-Means and how DBSCAN can overcome them?
    * What is DBSCAN and how to cluster geospatial data using DBSCAN?
<br>

### ASSOCIATION RULE LEARNING

An association rule learning problem is where you want to discover rules that describe large portions of your data, such as people that buy X also tend to buy Y.

9. [Apriori Algorithm](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/9.Apriori.ipynb)<br>
    Apriori algorithms is a data mining algorithm used for mining frequent itemsets and relevant association rules. It is devised to operate on a database that contains transactions -like, items bought by a customer in a store.

    In this lesson, we'll use [Market Basket Analysis](http://archive.ics.uci.edu/ml/machine-learning-databases/00352/) dataset to understand the following - 
    * What is Association rule?
    * What is Apriori algorithm and how to build it using mlxtend package?
<br>

### DIMENSIONALITY REDUCTION

Dimensionality reduction is used to find a better (less complex) representation of the data. After applying such a process, the data set should have a reduced amount of redundant information while the important parts may be emphasized. In practice, this could be realized as removing a column from a database from further analysis.

10. [Principal Component Analysis(PCA)](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/10.PCA.ipynb)<br>
    We will use [MNIST hand-written digit dataset](https://www.kaggle.com/c/digit-recognizer) to explore different techniques of dimensionality reduction. Principal Component Analysis (PCA) is fundamentally a simple dimensionality reduction technique that transforms the columns of a dataset into a new set features. It does this by finding a new set of directions (like X and Y axes) that explain the maximum variability in the data. We will also explore Linear Discriminant Analysis (LDA) and T-Distributed Stochastic Neighbour Embedding (t-SNE). 

    In this notebook you'll learn -
    * What is dimensionality reduction and its need?
    * What is PCA and how to build it from scratch? 
    * What is LDA and t-SNE and how to implement them using scikit-learn?
    * Compare the techniques using visualization. 
<br>

## NEURAL NETWORKS

11. [Neural Networks from scratch](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/11.NeuralNetworks.ipynb)<br>
    This course wouldn't be complete without understanding the fundamentals of Neural Networks. In this lesson, we will explore the concepts of Neural Netowrks using MNSIT handwritten dataset. 

    In this lesson, you'll learn -
    * What is Neural Networks?
    * What is feed forward propagation with an example? 
    * What is cost function and its derivation?
    * What is gradient descent?
    * What is Back propagation with an example?
    * How to implement Neural Networks from scratch using Numpy and Python on MNIST dataset?
<br>

## TIME-SERIES FORECASTING <br>
Time Series is a series of observations taken at specified time intervals usually equal intervals. Analysis of the series helps us to predict future values based on previous observed values. In Time series, we have only 2 variables, time & the variable we want to forecast.<br>

12. [AutoRegressive Integrated Moving Average (ARIMA)](https://github.com/NikhilSKashyap/11-Projects-to-DataScience/blob/master/12.ARIMA.ipynb)<br>
    ARIMA(Auto Regressive Integrated Moving Average) is a combination of 2 models AR(Auto Regressive) & MA(Moving Average). It has 3 hyperparameters - P(auto regressive lags),d(order of differentiation),Q(moving avg.) which respectively comes from the AR, I & MA components. The AR part is correlation between prev & current time periods. To smooth out the noise, the MA part is used. The I part binds together the AR & MA parts.<br>

    This notebook contains - 
    * ARIMA introduction
    * Decompose the Time series
    * Stationarize the data
    * Interpret ACF and PACF
    * Determine p, d, q
    * Forecast using ARIMA model
    * Forecast using Auto ARIMA