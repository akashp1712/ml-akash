---
title:  Fundamentals
has_children: false
nav_order: 2
---

# Fundamentals of Machine Learning

**1. Define Machine Learning**<br/>
Machine Learning is about building systems that can learn from data.<br/>
Learning means getting better at some task, given some performance measuermeasure.


**2. List out four types of problems that can be solved using it.**<br/>
Machine Learning is great for...<br/>
  - Problems for which existing solution require a lot of fine-tuning or long list of ruls.<br>
    One Machine Learning algorithm can often simplify code and perform better than the traditional approach.
  - **Complex problem** for which using a traditional approach yields no good solution.<br>
    The best Machine Learnign techniques can perhaps find a solution.<br/>
  - **Fluctuating environments**: A Machine Learning system can adapt to new data.<br/>
  - Getting insights about complex problems and large amounts of data.(e.g., data mining)


---
#### Types of Machine Learning Systems

There are so many different types of Machine Learning systems that it is useful to classify them in broad categories, based on the following criteria:<br/>

- Whether or not they are trained with human supervision<br/>
  (supervised, unsupervised, semisupervised and reinforcement learning)
- Whether or not they learn incrementally on the fly (online versus batch learning)
- Insance based(work by simply comparing new data points to known data points) OR Model based(work by detecting patterns in the training data and building a predictive model)<br/>

---


**3. What is a labeled training set?**<br/>
Labeled training set is a training set which contains the label(desired solution) for each instance.


**4. What are the two most common supervised tasks?**<br/>
The most common supervised tasks are **Classification** and **Regression.**


**5. What are the four most common unsupervised tasks?**<br/>
  - Clustering
  - Anomaly detection and novelty detection
  - Visualization and dimensionality reduction
  - Association rule learning


**6. What type of Machine Learning algorithms would you use to allow a robot to walk
   in various unknown terrains?**<br/>
Reinforcement Learning


**7. What type of algorithm would you use to segment your customers into multiple groups?**<br/>
If we can defien the lables then, classification algorithms(supervised learning) But if we can not define (or identify) the labels then, clustering algorithms (unsupervised learning)


**8. Spam detection is in example of Supervised learning Or Unsupervised learning?**<br/>
Spam detection is a supervised learning problem.


**9. What is an online learning system?**<br/>
In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches.
  - Learning is fast and cheap, so the system can learn about new data on the fly, as it arrives.
  - Great for system that receive data as continuous flow, And need to adapt to change rapidly.


**10. What is out-of-core learning?**<br/>
An Out-of-core learning algorithm splits the data into mini-batches and uses online learning(incremental) algorithms to learn from these mini-batches.This is useful when the entire data can not fit into computer's memory.


**11. What type of learning algorithm relies on a similarity measure to make predictions?**<br/>
Instance based learning algorithm relies on a similarity measure to make predictions.


**12. What is the difference between a model parameter and a learning algorithm's hyperparameter?**<br/>
A model has one or more *model parameter* that determines what it will predict given a new instance. A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances

A *hyperparameter* is a parameter of the learning algorithm itself, not of the model. (e.g., the amount of regularization to apply).


**13. What do model-based learnign algorithms search for? What is the most commmon strategy they use to succeed? How do they make predictions?**<br/>
 - Modle based learning algorithms search for an optimal value for the model parameter such that the model will generalize well to new instances.
 - The model gets trained by minimizing a cost function that measures how bad the system is at making predictions on the training data, plus a penalty for model complexity if the model is regularized.
 - The trained Modles make predictions by taking features of new instance and predicting using the learned function. 


**14. Can you name four main challenges in Machine Learning?**<br/>

#### Challenges wrt Bad Data

 - *Insufficient quantity of training data*: It takes a lot of data for most ML algorithms to work properly.
 - *Nonrepresentative Training Data*: IT is crucial to have training data that is representative of the new cases.
 ```
 If the sample is too small, you will have sampling noise, but even very large samples can be nonrepresentative if the sampling method is flawed. This is called sampling bias.
 ```
 - *Poor-Quality data*: If training data is full of errors, outliers, and noise, it will make harder for the system to detect the underlying patterns.
 ```
 Most data scientists spend a significant part of their time doing data cleaning only.
 ```
 - *Irrelevant Features*: System will only capable of learning if the training data contains enough relevant features and not too many irrelevant ones.
 ```
 Feature engineering: A critical part of the success of a Machine Learning project is coming up with a good set of features to train on.
 ```
 
#### Challenges wrt Bad Algorithms
 
 - *Overfitting the Training Data*: The model performs well on the training data, but it does not generalize well.
 - *Underfitting the Training Data*: The mode is too simple to learn the underlyting structure of the data.

**15. If your model performs great on the training data but generalizes poorly to new instances, What is happenning? Can you name three possible solutions?**<br/>
Here, The model is overfitt to the data. That happens when the model is too complex relative to the amount and noisiness of the training data. Followings are the possible solutions:<br/>

 -Simplifying the model by selecting one with fewer parameters, which is by choosing linear model over high-degree polynomial model(reduce the number of attributes in the training data).
 - Gather more training data.
 - Reduce the noise in the training data, like by fixing data errors and remove outliers.
 ```
 Constraining a model to make it simpler and reduce the risk of overfittign is called regularization.
 ```

**16. What is Underfitting? And How to fix it?**<br/>
Underfitting occurs when the model is too simple to learn the underlying structure of the data. Followings are the possible fix:<br/>
 - Select a more powerful model with more parameters.
 - Feed better features to the learning algorithm (feature engineering).
 - Reduce the constrain of the model (e.g., reduce the regularization hyperparameter).

    
**17. What is a test set, and why would you want to use it?**<br/>
We split the data into two sets: The *training set* and the *test set*. We train the model using the training set and test it using the test set.<br/>
We *evaluate* the model(error rate on new cases) using the test set which tell us how the model will perfomr on instances it has never seen before.


**18. What is the purpose of a validation set?**<br/>
We use holdout validation to hold out part of the training set to evaluate several candidates models and select the best one. The new hold-out set is called the validation set.<br/>

Here, how we can use it:
 - Holdout Validation: Train multiple models with various hyperparameters on the reduced training set (the full training set minus the validation set).
 - Train the best model on the full training set(including the validation set), which gives the final model.
 - Evaluate this final model on the test set to get an estimate of the generalize error.


**19. What is the train-dev set, when do you need it, and how do you use it?**<br/>
The *train-dev* set is used when there is risk of mismatch between the training data and the data used in the validation and test datasets.<br/>
The train-dev set is a part of the training data set that's held out. The model is trained on the rest of the training set, and evaluated on both the train-dev set and the validation set.
 - If the model performs well on the training set but not on the train-dev set, then the model is likely overfitting the training set.
 - If the model performs well on both the training set and the train-dev set, but not on the validation set, then there is probably a significant data mismatch between the *training data* and the *validation + test data*. And we should try to improve the trainig data to make it look like more like the validation + test data.
 

**20. What can go wrong if you tune hyperparameters using the test set?**<br/>
If you tune hyperparameters using the test set, you risk overfitting the test set.

