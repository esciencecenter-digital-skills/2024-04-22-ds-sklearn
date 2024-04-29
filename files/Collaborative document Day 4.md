![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 4

2024-04-22-ds-sklearn Machine learning in Python with scikit-learn.

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: https://tinyurl.com/sklearn-04-25

Collaborative Document day 1: https://tinyurl.com/sklearn-04-22

Collaborative Document day 2: https://tinyurl.com/sklearn-04-23

Collaborative Document day 3: https://tinyurl.com/sklearn-04-24

Collaborative Document day 4: https://tinyurl.com/sklearn-04-25

##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## 🎓 Certificate of attendance

If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl.
Please request your certificate within 8 months after the workshop, as we will delete all personal identifyable information after this period.

## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, raise your hand in zoom. Click on the icon labeled "Reactions" in the toolbar on the bottom center of your screen,
then click the button 'Raise Hand ✋'. For urgent questions, just unmute and speak up!

You can also ask questions or type 'I need help' in the chat window and helpers will try to help you.
Please note it is not necessary to monitor the chat - the helpers will make sure that relevant questions are addressed in a plenary way.
(By the way, off-topic questions will still be answered in the chat)


## 🖥 Workshop website

https://esciencecenter-digital-skills.github.io/2024-04-22-ds-sklearn/

🛠 Setup

https://esciencecenter-digital-skills.github.io/2024-04-22-ds-sklearn/#setup

## 👩‍🏫👩‍💻🎓 Instructors

Sven van der burg, Malte Luken, Johan Hidding

## 🧑‍🙋 Helpers

~~Claire Donnelly~~ Flavio Hafner

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

## 🗓️ Agenda
09:00	Welcome and icebreaker
09:15	Theory on selecting the best model: under & overfitting + learning curves
10:15	Coffee break
10:30	Pointers to advanced topics
11:00	Try out learned skills on Penguins dataset
11:30	Coffee break
12:30	Concluding remarks, Q&A
12:45	Wrap-up
13:00	END


## 🔧 Exercises

### Exercise: overfitting and underfitting 

#### 1: A model that is underfitting:

a) is too complex and thus highly flexible
b) is too constrained and thus limited by its expressivity
c) often makes prediction errors, even on training samples
d) focuses too much on noisy details of the training set

Select all answers that apply

#### 2: A model that is overfitting:

a) is too complex and thus highly flexible
b) is too constrained and thus limited by its expressivity
c) often makes prediction errors, even on training samples
d) focuses too much on noisy details of the training set

Select all answers that apply


Note: flexibility refers to training time. A model with many parameters is very flexible on the training data. But we could say it is less flexible to adopt to unsee data because it is overfitting. 



### Quiz: over- and underfitting and learning curves (5 minutes, in pairs; if time-permitting)

**1. A model is overfitting when:**
- a) both the train and test errors are high
- b) train error is low but test error is high
- c) train error is high but the test error is low
- d) both train and test errors are low

*select a single answer*

**2. Assuming that we have a dataset with little noise, a model is underfitting when:**
- a) both the train and test errors are high
- b) train error is low but test error is high
- c) train error is high but the test error is low
- d) both train and test errors are low

*select a single answer*


**3. For a fixed training set, by sequentially adding parameters to give more flexibility to the model, we are more likely to observe:**
- a) a wider difference between train and test errors
- b) a reduction in the difference between train and test errors
- c) an increased or steady train error
- d) a decrease in the train error

*Select all answers that apply*

**4. For a fixed choice of model parameters, if we increase the number of labeled observations in the training set, are we more likely to observe:**
- a) a wider difference between train and test errors
- b) a reduction in the difference between train and test errors
- c) an increased or steady train error
- d) a decrease in the train error

*Select all answers that apply*

**5. Polynomial models with a high degree parameter:**
- a) always have the best test error (but can be slow to train)
- b) underfit more than linear regression models
- c) get lower training error than lower degree polynomial models
- d) are more likely to overfit than lower degree polynomial models

*Select all answers that apply*

**6. If we chose the parameters of a model to get the best overfitting/underfitting tradeoff, we will always get a zero test error.**
- a) True
- b) False

*Select a single answer*

Group 1: 1B, 2A, 3AD, 4BC, 5CD, 6B
Group 2: 1b, 2a, 3ad, 4bd, 5cd, 6b
Group 3: 1B, 2C, 3AD, 4BC, 5CD, 6B
Group 4: 1B,2A,3(A,D),4(B,C),5(C,D),6B

#### Discussion of answers 

**4: why is (c) correct?**

for a given number of parameters, increasing the size of the training set increases or keeps the train error steady. the intuition is that starting from a model with as many parameters as training points, the error on the training set is 0. keeping fixed the parameters and increasing the size of the training set means that the model does not perfectly learn the training data anymore, thus increasing the training error.


### Exercise: Try out learned skills on penguins dataset
In this exercise we use the [Palmer penguins dataset](https://allisonhorst.github.io/palmerpenguins/)

We use this dataset in classification setting to predict the penguins’ species from anatomical information.

Each penguin is from one of the three following species: Adelie, Gentoo, and Chinstrap. See the illustration below depicting the three different penguin species:

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/palmer_penguins.png)

Your goal is to predict the species of penguin based on the available features. Start simple and step-by-step expand your approach to create better and better models.

![](https://carpentries-incubator.github.io/deep-learning-intro/fig/culmen_depth.png)

You can load the data as follows:
```python
penguins = pd.read_csv("../datasets/penguins_classification.csv")

```


**What is your approach, and why? How well does your model  do?**

Group 1: LogisticRegression (default): 0.961, regularization C = 0.5: 0.964, other settings result in lower cros-val-accuracy.
Group 2: K-nearest neighbours - 95.3%
Group 3: Dummy uniform/stratified: ~28-38%. Scaling + Logistic Regression: ~92-97%
Group 4: Scaling + model pipeline. With default settings: LogRegr: 0.959, RandFor 0.968, SVM: 0.959. Turning off scaling slightly improves results (LogRegr: 0.962, RandFor: 0.971, SVM: 0.965)


### Exercise: Validation and learning curves

#### Train and test SVM classifier (30 min,  2 people / breakout room)

The aim of this exercise is to:
* train and test a support vector machine classifier through cross-validation;
* study the effect of the parameter gamma (one of the parameters controlling under/over-fitting in SVM) using a validation curve;
* determine the usefulness of adding new samples in the dataset when building a classifier using a learning curve. 

We will use blood transfusion dataset located in `../datasets/blood_transfusion.csv`. First take a data exploration to get familiar with the data.

You can then start off by creating a predictive pipeline made of:

* a [`sklearn.preprocessing.StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) with default parameter;
* a [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

Script below will help you get started:

```python=
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(StandardScaler(), SVC())
```

You can vary gamma between 10e-3 and 10e2 by generating samples on a logarithmic scale with the help of

```python=
gammas = np.logspace(-3, 2, num=30)
param_name = "svc__gamma"
```

To manipulate training size you could use:

```python=
train_sizes = np.linspace(0.1, 1, num=10)
```


## 🧠 Collaborative Notes

### Under- & Overfitting 

- performance on training data != performance on unseen data 
- overfitting: model is very accurate on training data, but not on test data 
- underfitting: model is accurate neither on the training nor on the test data 
- polynomials??
    - degree 1 = straight line 
    - degree 2 = quadratic 
    - etc 
- N points can be fit with a degree N polynomial
- but the goal is to fit the data with a polynomial with degree k, where k is as small as possible but as large as necessary 
- a more complex model is more likely to fit to noise, which we want to avoid because it does not generalize to unseen data 
- need to find a balance between under- and over-fitting
- the complexity of the model interacts with the size of the data and the noise in the data 



### Comparing train and test errors

- varying complexity -> validation curves 
- varying sample size -> learning curves 
- **increasing complexity** increasing the number of degrees in the polynomial
    - train error will always go down 
    - test error will, at some point, start to increase
- we want to choose a model with the best test error
    - sweet spot between underfitting and overfitting 
- **varying the sample size**: learning curves 
    - how much better does the test error get as we increase the number of samples?
    - with few number of samples, adding more data improves the test error. this is because the model can extract more signal relative to noise 
    - as we increase the number of samples, the marginal gains of adding more samples decreases. this limit arises because of random noise in the data that the model cannot learn, no matter how large the number of samples
- this intuition applies to all models; but for different model families, model complexity means different things
    - for polynomials, it is their degree
    - for decision trees, it is the number of splits

### What to try next? 

- different models 
    - linear models / logistic regression 
    - support vector machines
    - decision trees
- support vector machines 
    - separate classes as good as possible with a hyperplane
    - both linear and non-linear hyperplanes
- ensemble learning
    - combine predictions from different models, for instance multiple decision trees
- neural networks / deep learning
    - can learn very flexible functions 
    - but need more training data
- deep learning is used for homogenous input (images, audio, text), whereas classical ML methods are used for heterogeneous data such as tabular data on persons
- **choosing best parameters and best model**
    - model sweep: run many models and select the one that performs best 
    - hyperparameter tuning: run the same model with different hyperparameters (for instance, different flexibility) and compare which combination of parameters works best 
    - one can also combine model sweep and hyperparameter tuning
- automatic feature selection 
    - start with many/all features, and then filter out those that are not useful for making the prediction, and keep those that are important for making the prediction
- explainable machine learning attempts to open the black box of machine learning and understand which aspects of the data contribute most to a certain prediction




## 📚 Resources
- fast.ai: crash course on deep learning 
- [dianna](https://github.com/dianna-ai/dianna): python library for explainable AI, developed by the eScience Center.
- link to post-workshop survey: https://www.surveymonkey.com/r/NSCGDGS
- [weights and biases](https://wandb.ai/site)
- [cookie cutter data science](https://drivendata.github.io/cookiecutter-data-science/)


