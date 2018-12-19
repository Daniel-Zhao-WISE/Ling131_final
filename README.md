##Movie Reviews Sentiment Classification
By Dongyu Zhao (dongyuzhao@brandeis.edu) <br>

To predict the sentiment polarity of a movie review, put that review into a file under
"data/positive" or "data/negative" directory. <br>

For the first time, it'll take about 1 hour to run to featured the data and train the
classifiers. All the accuracy result in the report will be print at console this time.
In addition, it'll create a cache file to save the featured, labeled data and the models. <br>

For the second time, it'll take about 1 minute to reload the data and model to predict the results.

---
### review_sentiment.py:
Core code of the review sentiment classification consists of two class
-- ReviewSentiment and Classifier. <br>

* ReviewSentiment: A class to store information of feature sets, train sets and test sets
    * Constructor <br>
        Required variable: <br>
        * labeled_data: whole dataset to train and test <br>
        
        Optional variables: <br>
        * train_size:size of the train_set, default one is 1000
        * features: list of feature sets, consists of "unigram", "bigrams",
        "unigram + bigrams", "unigram + POS", "adjective", "top # unigram".
        Default value is []
        * train_set: Training labeled, featured set, default value is [] 
        * test_set: Test labeled, featured set, default value is []
    * train_classifier(classifier, i, n) <br>
    Train a classifier with the train_set of , using n-fold cross validation
    and ith feature set in the object. ith feature set in the object. <br>
    Return a Classifiers object consists of n classifier corresponding to the n-fold. <br>
        * classifier: Classifier model for different algorithms <br>
        * i: index of the feature sets list (ith feature set to use) <br>
        * n: number of k-fold cross validation
    * evaluate_classifer(self, classifiers, i) <br>
    Evaluate classifiers on test_data. <br>
    Print an average accuracy.
    of the n-fold classifiers of ith feature set. 
        * classifiers: k-fold classifier models <br>
        * i: index of the feature sets list (ith feature set to use) 
    * predict(raw, classifiers, i) <br>
    Predict the classification results on a raw text of a movie review using 
    a random classifer in the k-fold classifier models with ith feature set. <br>
    Print the prediction.
        * raw: string of the raw text
        * classifiers: k-fold classifier models <br>
        * i: index of the feature sets list (ith feature set to use) 
    * save(dir) <br>
    Save this object's variable -- features, train_data, test_data using pickle
    to the directory.
        * dir: directory of where to save
    
* Classifiers: A k-fold classifier consists of k classifier model to be used in
the k-fold CV
    * Constructor
        * n: Number of the k-fold cross validation
    * classify(feature_set) <br>
    Randomly choose a classifier in the k-fold classifier to classify using. <br>
    Return a string of the classification result.
        * feature_set: feature_set used to generate features
---
### classification.py
Main executing part of training data and classification
* train(rs) <br>
Train the train_set in rs of different model (Naive Bayes or SVM) and different feature set
using 3-fold cross validation. <br>
Print the average accuracy result of the 3-fold cross validation on train_set.
    * rs: A ReviewSentiment object to store data and feature sets
* evaluate(rs, classifiers) <br>
Evaluate the test_set in rs of different model (Naive Bayes or SVM) and different feature set
using 3-fold cross validation. <br>
Print the average accuracy result of the 3-fold cross validation on test_set.
    * rs: A ReviewSentiment object to store data and feature sets
    * classifiers: A k-fold Classifiers object to classify
* predict(rs, dir, classifiers, i) <br>
Predict classification results of file in the directory using classifiers and ith
feature set to classify
    * rs: A ReviewSentiment object to store data and feature sets
    * dir: Directory of the data file to predict
    * classifiers: A k-fold Classifiers object to classify
    * i: index of the feature sets list (ith feature set to use)

---
### classification_small.py
This is a demo code for TA to run just for validation. <br>
In this demo, I limits the features' number of each feature set to reduce
the running time. <br>
However, the features' number is too small so that the model accuracy is very
poor.