![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 3

2024-04-22-ds-sklearn Machine learning in Python with scikit-learn.

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

This is the Document for today: https://tinyurl.com/sklearn-04-24

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

## 🗓️ Agenda
09:00	Welcome and icebreaker
09:15	Handling categorical data
10:15	Coffee break
10:30	Encoding categorical variables
11:15	Intuitions on tree-based models
11:30	Coffee break
11:45	Combining numerical and categorical data
12:45	Wrap-up
13:00	END

## 🛫 Start
- Icebreaker: What is something that not a lot of people know about you?
- Yesterday's feedback
- Questions from yesterday (think about the final exercise)




## 🔧 Exercises

### Ordinal encoding (5 minutes in pairs, then discussion): [Johan]

Q1: Is ordinal encoding appropriate for marital status? For which (other) categories in the adult census would it be appropriate? Why?
Q2: Can you think of another example of categorical data that is ordinal?
Q3: What problem arises if we use ordinal encoding on a sizing chart with options: XS, S, M, L, XL, XXL? (HINT: explore ordinal_encoder.categories_)
Q4: How could you solve this problem? (Look in documentation of OrdinalEncoder)
Q5: Can you think of an ordinally encoded variable that would not have this issue?

#### Answers:

Group 1: 1: No; workclass. 2: education grade, life phase   3: their order will not correspond to the size order 4: assign variables manually
Group 2: 3: default alphabetical assignment; 4: Use the 'category' parameter w/ an array (of categories in desired order)
Group 3: Q1: No. Q2: Income class. Q3: Categories will be encoded alphabetically and break the order. Q4: Pass the categories manually to argument `categories` when instantiating the encoder.
Group 4:Q1: no - maybe education by age group. Q2: some survey data (Likert scales). Q3: alphabetical
Group 5: Q1: No, Q2: Fever (no fever, slight fever, high fever, very high fever), Q3: 


### Exercise: The impact of using integer encoding for with logistic regression (groups of 2, 15min) [Johan]


Goal: understand the impact of arbitrary integer encoding for categorical variables with linear classification such as logistic regression.

We keep using the `adult_census` data set already loaded in the code before. Recall that `target` contains the variable we want to predict and `data` contains the features.

If you need to re-load the data, you can do it as follows:

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])
```


**Q0 Select columns containing strings**
Use `sklearn.compose.make_column_selector` to automatically select columns containing strings that correspond to categorical features in our dataset.

**Q1 Build a scikit-learn pipeline composed of an `OrdinalEncoder` and a `LogisticRegression` classifier**

You'll need the following, already loaded modules:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
```

Because OrdinalEncoder can raise errors if it sees an unknown category at prediction time, you can set the handle_unknown="use_encoded_value" and unknown_value parameters. You can refer to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) for more details regarding these parameters.


**Q2 Evaluate the model with cross-validation.**

You'll need the following, already loaded modules:

```python
from sklearn.model_selection import cross_validate

```

**Q3 Repeat the previous steps using an `OneHotEncoder` instead of an `OrdinalEncoder`**

You'll need the following, already loaded modules:

```python
from sklearn.preprocessing import OneHotEncoder

```

#### Answers

Group 1: Q2 (average) accuracy with Ordinal encoder=0.8, Q3 (average) accuracy with Onehot encoder=0.85
Group 2: Q2: Accuracy: ~0.75 with OrdinalEncoder, ~0.83 with OneHotEncoder
Group 3: 0.80 -Ordinal vs 0.87 -OneHot
Group 4: Q2: 0.755; Q3: 0.832
Group 5: Q2: accuracy ~0.75, Q3: accuracy ~0.83


## 🧠 Collaborative Notes

### Handling Categorical Data


First we are removing the education-num column as usually we would not have that kind of information in raw data. Then separate out the target from the rest of the data.

```python
import pandas as pd
```

```python
adult_census = pd.read_csv("../datasets/adult-census.csv")
adult_census = adult_census.drop(columns="education-num")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name])
```

We can check which data is categorical and which is numerical by typing:

```python
data.dtypes
```

Select columns that are categorical:
```python
from sklearn.compose import make_column_selector as selector
```

```python
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
```

```python
data_categorical = data[categorical_columns]
data_categorical.head()
```

We need to turn categories into numbers, which is called encoding the data. 
We will cover 2 types of encoding: ordinal and one-hot. 

#### Ordinal Encoding

```python
from sklearn_preprocessing import OrdinalEncoder

education_column = data_categorical[["education"]]
encoder = OrdinalEncoder().set_output(transform="pandas")
education_encoded = encoder.fit_transform(education_column)
education_encoded
```

```python
data_encoded = encoder.fit_transform(data_categorical)
data_encoded.head()
```

Ordinal encoding can create a problem where it can introduce numerical ordering of the data, which may not reflect the true relationships between the categories. If running a regression for example, you may find that there are categories next to eachother that are not related. 

#### One-Hot Encoding

We can repeat the same process, this time using the one-hot encoder:
```python
from sklearn.preprocessing import OneHotEncoder

education_column = data_categorical[["education"]]
encoder = OneHotEncoder(sparse_output = False).set_output(transform="pandas")
education_encoded = encoder.fit_transform(education_column)
education_encoded
```

With one-hot encoding every unique value in a category is turned into a column. This can lead to increased dimensionality and the model could become very large.

```python
data_encoded = encoder.fit_transform(data_categorical)
data_encoded.head()
```

When should you use one-hot encoding instead of ordinal? 
 - Depends on the kind of learning you want to do, for some it will not matter. If you are doing linear models, then one-hot is usually the way to go. If you have trees, then either is fine. 


### Decision Trees
#### For regression and classification

- Sequence of simple decision rules
    - one feature and one threshold at a time
- They are good at replicating non-linear patterns
- No scaling required (unlike linear models)
- They are not that powerful on their own. Mostly useful as a building block for ensemble models, such as:
    - Random Forests
    - Gradient Boosting Decision Trees 

### Combining numerical and categorical data

If you don't have it in your notebook, reload the data and separate out the target from the data.

```python
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
adult_census = adult_census.drop(columns="education-num")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name])
```

#### Using the ColumnTransformer

Similarly to earlier where we selected categorical data, we now separate out the categorical and numerical data.

```python
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude = "object")
categorical_columns_selector = selector(dtype_include = "object")

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)
```

We now use the one-hot encoding on the categorical data and the standard scaler on the numerical data. 

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(handle_unknown = "ignore")
numerical_preprocessor = StandardScaler()
```

Apply the ColumnTransformer:

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", categorical_preprocessor, categorical_columns),
        ("standard_scaler", numerical_preprocessor, numerical_columns)
    ]
)
```

What the ColumnTransformer does: 

- Splits columns in original dataset based on names or indices provided.
- Transforms each subsets and then concatonates the subsets into a single one.

![](https://inria.github.io/scikit-learn-mooc/_images/api_diagram-columntransformer.svg)

#### Using the ColumnTransformer in a machine learning pipeline

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, RandomForestClassifier())
model
```

Before we can fit our data to the pipeline, we need to split our data into a training and testing dataset. For simplicity, we are not doing cross-validation.

```python
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state = 42
)
```
Now we can fit it:

```python
_ = model.fit(data_train, target_train)
```

We can then make predictions on new data:

```python
model.predict(data_test[:5])
```
 & predict on the test data and compare these to the expected test labels to compute the accuracy.

```python
model.score(data_test, target_test)
```

For cross-validation with the pipeline:

```python
from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=5)
cv_results
```

Changing the model you use, does not change how the pipeline works. So you can test multiple models (by changing one line of code) to see which one works best for your data. 


## 🎤Feedback
Please give us some (anonymous) feedback about day 3 of the workshop.
Think of the pace, instructors, content, exercises. Do you want more breakout rooms, more theory, more live coding?



## 📚 Resources
- [Today's lesson material](https://inria.github.io/scikit-learn-mooc/predictive_modeling_pipeline/03_categorical_pipeline_index.html)
