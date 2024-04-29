![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 1

2024-04-22-ds-sklearn Machine learning in Python with scikit-learn.

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: https://tinyurl.com/sklearn-04-22

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

Claire Donnelly

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city


## 🥶 Icebreaker
If you could be an animal for one day, which animal would you pick?


## 🗓️ Agenda
09:00   Welcome and icebreaker
09:15	Machine learning concepts
10:15	Coffee break
10:30	Tabular data exploration
11:30	Coffee break
11:45	First model with scikit-learn
12:45	Wrap-up
13:00	END

## 🧙‍♀️What to expect from this course?
* We can only cover the basics
* At the same time highlight state of the art practices
* And answer questions from your own practice

## 🔧 Exercises

### Exercise: Machine learning concepts
Given a case study: pricing apartments based on a real estate website. We have thousands of house descriptions with their price. Typically, an example of a house description is the following:

“Great for entertaining: spacious, updated 2 bedroom, 1 bathroom apartment in Lakeview, 97630. The house will be available from May 1st. Close to nightlife with private backyard. Price ~$1,000,000.”

We are interested in predicting house prices from their description. One potential use case for this would be, as a buyer, to find houses that are cheap compared to their market value.

#### What kind of problem is it?

a) a supervised problem
b) an unsupervised problem
c) a classification problem
d) a regression problem

Solution: AD

Select all answers that apply

#### What are the features?

a) the number of rooms might be a feature
b) the post code of the house might be a feature
c) the price of the house might be a feature

solution: AB

Select all answers that apply

#### What is the target variable?

a) the full text description is the target
b) the price of the house is the target
c) only house description with no price mentioned are the target

solution: B

Select a single answer

#### What is a sample?

a) each house description is a sample
b) each house price is a sample
c) each kind of description (as the house size) is a sample

Solution: A

Select a single answer

#### (optional) Think of a machine learning task that would be relevant to solve in your research field. Try to answer the above questions for it.



### Exercise: Data exploration (15min,  in groups) []

Imagine we are interested in predicting penguins species based on two of their body measurements: culmen length and culmen depth. First we want to do some data exploration to get a feel for the data.

The data is located in `../datasets/penguins_classification.csv`.

Load the data with Python and try to answer the following questions:
1. How many features are numerical? How many features are categorical?
2. What are the different penguins species available in the dataset and how many samples of each species are there?
3. Plot histograms for the numerical features
4. Plot features distribution for each class (Hint: use `seaborn.pairplot`).
5. Looking at the distributions you got, how hard do you think it will be to classify the penguins only using "culmen depth" and "culmen length"?
6. (optional): Look at `bike-rides.csv`

- Group 1:
    - 1: 2 numerical, 1 categorical
    - 2: Adelie, 151; Gentoo, 123; Chinstrap, 68
    - 
- Group 1/2:
    - 1: 2xnum, 1xcat
    - 2: Adelie: 151, Gentoo: 123, Chinstrap: 68
    - 3: done
    - 4: done
    - 5: On the basis of 1 feature, some overlap - on the basis of both features, it may be possible
- Group 3:
    - 1. 2 features are numerical and 1 feature is categorical
    - 2. Adelie 151; Gentoo 123; Chinstrap 68
    - 3. 
    - 4. 
- Group 4:
- Group 5:
    - 1: 2 numerical, 1 categorical
    - 2: Adelie 151, Gentoo 123, Chinstrap 68
    - 3: OK
    - 4: OK
    - 5: Adelie vs. Gentoo: yes. Chinstrap can only be distinguished on the combination of length and depth

### 📝 Exercise : Adapting your first model
The goal of this exercise is to fit a similar model as we just did to get familiar with manipulating scikit-learn objects and in particular the `.fit/.predict/.score` API.

Before we used `model = KNeighborsClassifier()`. All scikit-learn models can be created without arguments. This is convenient because it means that you don’t need to understand the full details of a model before starting to use it.

One of the KNeighborsClassifier parameters is n_neighbors. It controls the number of neighbors we are going to use to make a prediction for a new data point.

#### 1. What is the default value of the n_neighbors parameter? 
Hint: Look at the documentation on the scikit-learn website or directly access the description inside your notebook by running the following cell. This will open a pager pointing to the documentation.
```python
from sklearn.neighbors import KNeighborsClassifier

KNeighborsClassifier?
```
**Correct answer: 5**

#### 2. Create a KNeighborsClassifier model with n_neighbors=50
a. Fit this model on the data and target loaded above
b. Use your model to make predictions on the first 10 data points inside the data. Do they match the actual target values?
c. Compute the accuracy on the training data.
d. Now load the test data from "../datasets/adult-census-numeric-test.csv" and compute the accuracy on the test data.
**Correct answer: You should see a small improvement**


#### 3. (Optional) Find the optimal n_neighbors
What is the optimal number of neighbors to fit a K-neighbors classifier on this dataset?

- room 1: 1. 5; 3. accu = 0.825 when k = 8, flat after 10.
- room 2: Only small improvement in accuracy when n_neighbors changed from 5 to 50. Score is around 82-83% always.
![](https://codimd.carpentries.org/uploads/upload_1d53589c2c29898b3036da7a9eed6daa.png)

- room 3: 1: 5, score very similar (0.829)
- room 4: 1=5, 2b=9/10 match, 2c=82.9%, 2d=81.9%
- room 5: n_neighbors=8

## 🧠 Collaborative Notes

### Introduction Machine Learning

- Prediction on unseen data based on existing data:
    - classification
    - prediction (or short-term extrapolation)
    - extension on "classical" statistics

If we were to test our model on the original data, we would expect no errors. There is a difference between **memorizing** and **generalizing**, see also: over-fitting. We make a careful distinction between **training data** and **test data** to see how good our model is.

Usually our data is presented in a table, where the columns represent features or descriptors, and the rows are different individual observations.

In **supervised machine learning** there is a **ground truth** available that we aim for. Example: classification

In **unsupervised machine learning** we try to learn general features in the data without refering to some desired outcome. Example: cluster analysis

Elementary techniques:
- Linear models: regression
- Decision trees
- Support Vector Machines (SVM)

More advanced:
- Ensemble learning: examples: gradient boosted trees or random forrest.
- Neural networks: see also our workshop on **deep learning**!

To make the distinction:

$$\text{AI} \supset \text{Machine Learning} \supset \text{Deep Learning}$$

AI is an entire research area, ML is a collection of techniques, DL is one family of these.

### Data exploration (Johan)

#### Shortcuts in jupyter lab:
- SHIFT + TAB (when inside parentheses): to show documentation for a function
- SHIFT + ENTER (when inside a cell): execute cell


#### Read in the data:

```python
import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census.csv")
adult_census.head() # Look at the first 5 rows of the dataset
```

#### Explore the data
See how many samples for each class we have:
```python
target_column = "class"
adult_census[target_column].value_counts()
```

See how many samples in each category for workclass feature
```python
adult_census["workclass"].value_counts()
```

Plot the numerical data:
```python
adult_census.hist(figsize=(20, 14))
```

Create a crosstab for the two education features:
```python
pd.crosstab(index=adult_census["education"],
            columns=adult_census["education-num"])
```

Make a pairplot, showing how 3 features relate to each other:
```python
import seaborn as sns
n_samples_to_plot = 5000
columns = ["age", "education-num", "hours-per-week"]
sns.pairplot(
    data=adult_census[:n_samples_to_plot],
    vars=columns,
    plot_kws={"alpha": 0.2},
    hue=target_column,
    diag_kind="hist",
    diag_kws={"bins": 30}
    )
```

### First machine learning model
Load in the dataset
```python
import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census-numeric.csv")
```

Quick look at the data:
```
adult_census.head()
```

Separate the training data (features) and the target:
```python
target_name = "class"
target = adult_census[target_name]
```
```python
data = adult_census.drop(columns=[target_name])
```

See which columns are in the data:
```python
data.columns
```

Inspect data size:
```
data.shape[0] # number of samples
data.shape[1] # number of features
```

### Train a model
K-nearest neighbors is a machine learning algorithm that classifies or predicts new data points based on their proximity to existing labeled data points.

```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
_ = model.fit(data, target)
```

Make predictions
```python
target_predicted = model.predict(data)
target_predicted[:5]
target[:5] # ground truth
```

Evaluate predictions
```python
target[:5] == target_predicted[:5]
(target == target_predicted).mean() #compute accuracy
```

Load the test data:
```python
adult_census_test = pd.read_csv("../datasets/adult-census-numeric-test.csv")
adult_census_test.head()
```

Prepare test data:
```python
target_test = adult_census_test[target_name]
data_test = adult_census_test.drop(columns=[target_name])
```

Evaluate model on test dataset:
```python
accuracy = model.score(data_test, target_test)
accuracy
```


## 🎤Feedback
Please give us some (anonymous) feedback about day 1 of the workshop.
Think of the pace, instructors, content, exercises. Do you want more breakout rooms, more theory, more live coding?




## 📚 Resources
- [Upcoming workshops](https://www.esciencecenter.nl/events/?f=workshops)
- [Subscribe to our newsletter](eepurl.com/dtjzwP)
- [Dataspell: an IDE that extra features for data exploration](https://www.jetbrains.com/dataspell/)

