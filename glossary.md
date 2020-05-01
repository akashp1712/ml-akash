---
title:  Glossary
has_children: false
nav_order: 4
---

# Machine Learning Glossary
---

- **Anomaly detection**<br/>
The system is shown mostly normal instances during training, os it learns to recognize them.<br/>
Then, when it sees a new instance, it can tell whether it looks like a normal one or whether it is likely an anomaly.


- **Assoicate Rule learning**<br/>
The goal is to dig into large amounts of data and discover interesting relations between attributes.


- **Cost function**<br/>
Cost function measures how bad your model is.


- **Dimensionality reduction**<br/>
It is a task that simplifies the data without loosing too much information.<br/>
e.g., One wasy to do this is to merge several correlated features into one.


- **Feature Engineering**<br/>
Feature engineering consists the follwoing steps:
   - Feature selection(selecting the most useful features to train on among existing features)
   - Feature extraction(combining existing features to produce a more useful on)
   - Creating new features by gathering new data
 
 
- **Feature extraction**<br/>
Feature extraction is identifying new feature from the existing set of features.<br/>
e.g., By combining multiple features into one using dimensionality reduction.


- **Feature Scaling**<br/>
Practice of making all attributes to have the same scale.
    - **Normalization(min-max scaling):** Values are shifted and rescaled so they end up ranging from 0 to 1. It substracts the min value and divides it by the max minus the min.
    - **Standardization:** It substracts the mean value and then it divides by the standard deviation so that the resultiing distribution has unit variance. Standardization doesn't bound valeus to a specific range(i.e, 0 to 1). However, Standardization is much less affected by outliers.


- **Fitness function**<br/>
Fitness function or Utility function measures how good your model is.


- **Generalization error**<br/>
The error rate on new cases is called the generalization error (or out-of-sample error).<br/>
If the training error is low(i.e., the model makes few mistaks on the test set) but the generalization error is high, it means that your model is overfitting the training data.


- **Hierarchical clusterin algorithm**<br/>
It is an Unsupervised learning technique, It subdivides each group into smaller group.<br/>


- **Inference**<br/>
Inference is applying the model to make predictions on new cases.


- **Instance-based learning**<br/>
In this kind of learning, The system learns the example, then generalizes to new cases by using a similarity measure to compare them to the learned examples.


- **Novelty detection**<br/>
It aims to detect new instances that look different from all instances in the training set.


- **Offline learning(Batch learning)**<br/>
In batch learning, system is incapable of learning incrementally: it must be trained using all the available data. First the system is trained, and then it is launched into production and runs wihout learning anymore. It just applies what is has learned.


- **Online learning**<br/>
In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches.


- **Overfitting**<br/>
It means that the model performs well on the training data, but it does not generalize well.<br/>
Overfitting happens when the model is too complex relative to the amount and noisiness of the training data.


- **Regularization**<br/>
    - Constraining a model to make it simpler and reduce the risk of overfittign is called regularization.
    - The amout of regularization to apply during learning can be controlled by a hyperparameter. Is it aparameter of a learning algorithm and must be set prior to training and remains constant during training.
 
 
- **RMSE**<br/>
    - Root Mean Square Error is generally the preferred performance measure for regression tasks.
    - It gives and idea of how much error the system typically makes in its predictions, with higher weight for large errors.

- **Underfitting**<br/>
Underfitting happens when the model is too simple to learn the underlying structure of the data.


- **Visualization Algorithms**<br/>
They are example of unsupervised learning, They take lots of complex and unlabeled data<br/>
And they output a 2D or 3D representation of your data that can be easily plotted.

