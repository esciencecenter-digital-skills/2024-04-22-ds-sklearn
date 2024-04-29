![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 2

2024-04-22-ds-sklearn Machine learning in Python with scikit-learn.

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: https://tinyurl.com/sklearn-04-23

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

Hugo: hi all, unfortunately I have a bit of a flu, so I cannot join today. I hope to be able to join tomorrow again (couldn't find contact details so posting it here).
Sven: Sad to hear that Hugo! Hope you get better soon :) You can always contact us at training@esciencecenter.nl
Hugo: Many thanks, Sven! And thanks for the contact address.

## 🗓️ Agenda
09:00	Welcome and icebreaker
09:15	Working with numerical data + intuitions on linear models
10:15	Coffee break
10:30	Preprocessing features for numerical features
11:30	Coffee break
11:45	Model evaluation using cross-validation
12:45	Wrap-up
13:00	END

## Welcome
- Feedback from yesterday
- Questions from yesterday
- Icebreaker: desk yoga

## 🔧 Exercises

### Exercise: Compare with simple baselines [Sven]
#### 1. Compare with simple baseline
The goal of this exercise is to compare the performance of our classifier in the previous notebook (roughly 81% accuracy with LogisticRegression) to some simple baseline classifiers. The simplest baseline classifier is one that always predicts the same class, irrespective of the input data.

What would be the score of a model that always predicts ' >50K'?

What would be the score of a model that always predicts ' <=50K'?

Is 81% or 82% accuracy a good score for this problem?

Use a DummyClassifier such that the resulting classifier will always predict the class ' >50K'. What is the accuracy score on the test set? Repeat the experiment by always predicting the class ' <=50K'.

Hint: you can set the strategy parameter of the DummyClassifier to achieve the desired behavior.

You can import DummyClassifier like this:
```python
from sklearn.dummy import DummyClassifier
```

#### 2. (optional) Try out other baselines
What other baselines can you think of? How well do they perform?

- group 1: >50K=23.4%, <=50K=76.6%.
- group 2: (<50k, >50k): (0.766, 0.234) depends on target test set. prior/most_frequent/constant: 0.766; stratified: 0.641; uniform: 0.497. When using constant, classes have an empty space at the beginning e.g. " >50K".
- group 3: Always predict >50K: 0.234; always predict <=50K: 0.766; Other baseline: random guess, most_frequent (score: 0.766), stratified/random (score: 0.640)

### Exercise: Recap fitting a scikit-learn model on numerical data
#### 1. Why do we need two sets: a train set and a test set?

a) to train the model faster
b) to validate the model on unseen data [CORRECT]
c) to improve the accuracy of the model

Select all answers that apply

#### 2. The generalization performance of a scikit-learn model can be evaluated by:

a) calling fit to train the model on the training set, predict on the test set to get the predictions, and compute the score by passing the predictions and the true target values to some metric function [CORRECT]
b) calling fit to train the model on the training set and score to compute the score on the test set [CORRECT]
c) calling cross_validate by passing the model, the data and the target [CORRECT]
d) calling fit_transform on the data and then score to compute the score on the test set

Select all answers that apply

#### 3. When calling `cross_validate(estimator, X, y, cv=5)`, the following happens:

a) X and y are internally split five times with non-overlapping test sets [CORRECT]
b) estimator.fit is called 5 times on the full X and y
c) estimator.fit is called 5 times, each time on a different training set [CORRECT]
d) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the train sets
e) a Python dictionary is returned containing a key/value containing a NumPy array with 5 scores computed on the test sets [CORRECT]

Select all answers that apply

#### 4. (optional) Scaling
We define a 2-dimensional dataset represented graphically as follows:
![](https://i.imgur.com/muvSbI6.png)

Question

If we process the dataset using a StandardScaler with the default parameters, which of the following results do you expect:

![](https://i.imgur.com/t5mTlVG.png)


a) Preprocessing A [CORRECT]
b) Preprocessing B 
c) Preprocessing C
d) Preprocessing D

Select a single answer

#### 5. (optional) Cross-validation allows us to:

a) train the model faster
b) measure the generalization performance of the model [CORRECT]
c) reach better generalization performance
d) estimate the variability of the generalization score [CORRECT]

Select all answers that apply



## 🧠 Collaborative Notes

### Working with numerical data

Recap: Loading the US Census data set.

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
# We drop the column education-num 
adult_census = adult_census.drop(columns="education-num")
adult_census.head()
```

Recap: Separate data and target.

```python
data, target = adult_census.drop(columns="class"), adult_census["class"]
```

Selecting numerical features.

```python
# Print data type of each column
data.dtypes
# Select columns with `int64` type
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
# Select data with numerical columns
data_numeric = data[numerical_columns]
data_numeric.head()
```

Excourse: Handling missing data
- Missing data is commonly represented as `NaN`
- Should be dealt with before entering into a machine learning model
- Missing values can be *inputed* using for example the mean of the column (note that this requires certain assumptions to be met which go beyond this course)
- Feed a variable into the machine learning model indicating whether another variable is missing or not

Train-test-splits

```python
from sklearn.model_selection import train_test_split

# Split data set in 75% train and 25% test set
data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target,
    # Relative size of test set
    test_size=0.25,
    # Allows exactly reproducing the split, othwerise different split each time
    random_state=42
)

# Print numbers rows and columns of train and test set
data_train.shape, data_test.shape

```

Important points:
- The test set should be independent from train set, e.g., if decisions on training a model are based on performance on test set, the test set becomes "dirty", and is not as informative on generalization performance anymore (it might be a good idea to get a new test set).
- The size of the test set should depend on the improvement in accuracy you are aiming for. Smaller improvements require larger test sets.

### Intuition on linear models

- Linear combination of features plus intercept (offset, bias)

For regression:
- Fit a straight line through data points by minimizing the sum of squared differences between the line and each data point
- For more than one feature, the line becomes a multidimensional plane

For classification with two target classes: Logistic regression
- Fit a logistic function to target, that is, the probability of the target being one of two labels given the features
- Rephrased: Fit a linear decision boundary that best separates two classes

For more than two target classes: Multinomial regression
- Fit multiple decision boundaries that best separate the classes

Non-linearly separable targets:
- Require feature engineering (transforming the features)
- Or a more complex model

Conclusion:
- Linear models are simple and fast
- Can perform poorly when there are much fewer features than data points
- Work well for many features

### Fitting a linear model

```python
from sklearn.linear_model 
LogisticRegression

model = LogisticRegression()

# Train model on train data
model.fit(data_train, target_train)

# Calculate accuracy on test data
accuracy = model.score(data_test, target_test)

```

## Preprocessing numerical values
### Load the data
```python
import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census.csv")
```

### Prepare data
```python
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)
```

#### Use only numerical features
```python
numerical_columns = ["age", "capital-gain",
                     "capital-loss", "hours-per-week"]
data_numeric = data[numerical_columns]
```

#### Split data in train and test set
```python
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42)
data_train.shape, data_test.shape
```

### Data preprocessing
Describe the numerical data:
```python
data_train.describe()
```

#### Scaling numerical features
Import and instantiate the standard scaler
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Fit the scaler to the data:
```python
scaler.fit(data_train)
```
Checkout the learned mean and scale/standard deviation:
```python
scaler.mean_
scaler.scale_
```

Transform the training data
```python
data_train_scaled = scaler.transform(data_train)
data_train_scaled
```

Fit and transform training data in one step
```python
data_train_scaled = scaler.fit_transform(data_train)
```

Use a standard scaler that has output as pandas
```python
scaler = StandardScaler().set_output(transform="pandas")
data_train_scaled = scaler.fit_transform(data_train)
data_train_scaled.describe()
```

Code for plotting:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# number of points to visualize to have a clearer plot
num_points_to_plot = 300

sns.jointplot(
    data=data_train[:num_points_to_plot],
    x="age",
    y="hours-per-week",
    marginal_kws=dict(bins=15),
)
plt.suptitle(
    "Jointplot of 'age' vs 'hours-per-week' \nbefore StandardScaler", y=1.1
)

sns.jointplot(
    data=data_train_scaled[:num_points_to_plot],
    x="age",
    y="hours-per-week",
    marginal_kws=dict(bins=15),
)
_ = plt.suptitle(
    "Jointplot of 'age' vs 'hours-per-week' \nafter StandardScaler", y=1.1
)
```

Create a pipeline of a standard scaler and logistic regression model
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
```
```python
model = make_pipeline(StandardScaler(),
                      LogisticRegression())
model
```
Look at the names of the different steps
```python
model.named_steps
```

Fit the pipeline:
```python
model.fit(data_train, target_train)
```
During training this is what happens in the pipeline
![](https://codimd.carpentries.org/uploads/upload_d6e121814bc51d194d9d7f6b26d378da.png)

Make predictions
```python
predicted_target = model.predict(data_test)
predicted_target[:5]
```
During prediction this is what happens:
![](https://codimd.carpentries.org/uploads/upload_5dcece0822553ab336dc21b69b0362e1.png)

Evaluate model:
```python
model.score(data_test, target_test)
```

Data leakage: test and training set are not independent anymore. Information from training set will 'leak' into test set. For example if you fit a scaler on the full dataset instead of just the training dataset.

### Model evaluation using cross-validation
Loading in data, prepare data (to copy-paste)
```python
import pandas as pd
adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)
numerical_columns = ["age", "capital-gain",
                     "capital-loss", "hours-per-week"]
data_numeric = data[numerical_columns]
```

Create a machine learning pipeline
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())
```

#### Cross-validation
![](https://codimd.carpentries.org/uploads/upload_3b79892895445397610537f7aff0103c.png)

Use crossvalidation to validate our model:
```python 
from sklearn.model_selection import cross_validate
cv_result = cross_validate(model, data_numeric, target,
                           cv=5)
cv_result
```

Summarize the cross-validation results
scores = cv_result['test_score']
```python
scores = cv_result['test_score']
scores.mean(), scores.std()
```

## 🎤Feedback
Please give us some (anonymous) feedback about day 2 of the workshop.
Think of the pace, instructors, content, exercises. Do you want more breakout rooms, more theory, more live coding?

## 📚 Resources
- [Machine learning course (deep understanding of machine learning algorithms and the math behind it) - Andrew Ng](https://www.coursera.org/specializations/machine-learning-introduction)
- [Lesson material, which we don't follow 1-on-1, but can be nice to read up on what we did so far](https://esciencecenter-digital-skills.github.io/scikit-learn-mooc/)

