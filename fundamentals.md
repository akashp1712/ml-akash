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
    
**14. Can you name four main challenges in Machine Learning?**<br/>

**15. If your model performs great on the training data but generalizes poorly to new instances, What is happenning? Can you name three possible solutions?**<br/>
    
**16. What is a test set, and why would you want to use it?**<br/>

**17. What is the purpose of a validation set?**<br/>

**18. What is the train-dev set, when do you need it, and how do you use it?**<br/>

**19. What can go wrong if you tune hyperparameters using the test set?**<br/>


