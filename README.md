#Pokemon Battle Outcome Prediction

# Introduction
This project involves predicting the outcomes of Pokemon battles using machine learning techniques. The process includes:
- Data Preprocessing: Cleaning and transforming the dataset for analysis.
- Exploratory Data Analysis (EDA): Visualizing and understanding the data.
- Model Building and Training: Creating and training a machine learning model.
- Model Evaluation: Assessing the performance of the model.


# Setting Up the Environment
## Installation of Necessary Libraries
Ensure you have the following libraries installed:

## Importing the Required Libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import FeatureHasher
```


# Data Loading and Initial Inspection
## Loading the Datasets
```
pokemon = pd.read_csv("pokemon.csv")  # Load Pokemon Dataset
combats = pd.read_csv("combats.csv")  # Load Combats Dataset
```

## Handling Missing Values and Encoding
```
pokemon["Type 2"] = pokemon["Type 2"].fillna("NA")
pokemon["Legendary"] = pokemon["Legendary"].astype(int)

h1 = FeatureHasher(n_features=5, input_type='string')
h2 = FeatureHasher(n_features=5, input_type='string')
d1 = h1.fit_transform(pokemon["Type 1"])
d2 = h2.fit_transform(pokemon["Type 2"])

d1 = pd.DataFrame(data=d1.toarray())
d2 = pd.DataFrame(data=d2.toarray())

pokemon = pokemon.drop(columns=["Type 1", "Type 2"])
pokemon = pd.concat([pokemon, d1, d2], axis=1)
```


# EDA
## Unique Values in the Dataset
```
print(pokemon.nunique())
```

## Visualizing Pokemon Types
```
ax = pokemon['Type 1'].value_counts().plot(kind='bar', color="yellow", edgecolor="red", figsize=(14, 8), title="Number of Pokemons based on their Type 1")
ax.set_xlabel("Pokemon Type 1")
ax.set_ylabel("Frequency")
plt.show()

ax = pokemon['Type 2'].value_counts().plot(kind='bar', color="thistle", edgecolor="indigo", figsize=(14, 8), title="Number of Pokemons based on their Type 2")
ax.set_xlabel("Pokemon Type 2")
ax.set_ylabel("Frequency")
plt.show()
```

## Total Number of Pokemons
```
number = pokemon["#"]
print('Total number of Pokemons is', len(number))
```

## Legendary Pokemon Rate
```
Legendary = pokemon["Legendary"]
rate = np.mean(Legendary == True)
print('Legendary rate=', rate)
```

## Correlation Heatmap
```
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(pokemon.corr(), ax=ax, annot=True, linewidths=0.05, fmt='.2f', cmap="magma")
plt.show()
```

## Pokemon Attack vs. Defense
```
df2 = pokemon.loc[:, ["Attack", "Defense"]]
df2.plot()
plt.show()
```

## Generation Distribution
```
generation = dict(pokemon['Generation'].value_counts())
gen_counts = generation.values()
gen = generation.keys()

fig = plt.figure(figsize=(8, 6))
fig.suptitle("Percentage of Generation based Distribution of Pokemon")
ax = fig.add_axes([0, 0, 1, 1])
explode = (0.1, 0, 0, 0, 0, 0)
ax.axis('equal')
plt.pie(gen_counts, labels=gen, autopct='%1.2f%%', shadow=True, explode=explode)
plt.show()
```

## Legendary Pokemon Distribution
```
generation = dict(pokemon['Legendary'].value_counts())
gen_counts = generation.values()
gen = generation.keys()

fig = plt.figure(figsize=(8, 6))
fig.suptitle("Percentage of Legendary Pokemon in Dataset (False: Not Legendary, True: Legendary)")
ax = fig.add_axes([0, 0, 1, 1])
explode = (0.2, 0)
ax.axis('equal')
plt.pie(gen_counts, labels=gen, autopct='%1.2f%%', shadow=True, explode=explode)
plt.show()
```


# Model Building and Training
## Preparing the Battle Data
```
data = []
for t in combats.itertuples():
    first_pokemon = t[1]
    second_pokemon = t[2]
    winner = t[3]
    
    x = pokemon.loc[pokemon["#"]==first_pokemon].values[:, 2:][0]
    y = pokemon.loc[pokemon["#"]==second_pokemon].values[:, 2:][0]
    z = np.concatenate((x, y))
    
    if winner == first_pokemon:
        z = np.append(z, [0])
    else:
        z = np.append(z, [1])
        
    data.append(z)

data = np.asarray(data)
X = data[:, :-1].astype(int)
y = data[:, -1].astype(int)
```

## Splitting Data into Training and Testing Sets and then training the model
```
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)

clf = RandomForestClassifier(n_estimators=100)
model = clf.fit(train_x, train_y)
pred = model.predict(test_x)
```


# Model Evaluation
## Accuracy and Classification Report
```
print('Accuracy:', accuracy_score(pred, test_y))
print(classification_report(test_y, pred))
```


# Conclusion
This project demonstrates how to predict the outcomes of Pokemon battles using a Random Forest Classifier. By preprocessing the data, performing exploratory data analysis, building, and evaluating the model, we achieved a high accuracy of around 95%.


# Future Scope
- Advanced Models: Experimenting with more advanced models like XGBoost or deep learning architectures.
- Feature Engineering: Creating more features from the existing dataset to improve model performance.
- Hyperparameter Tuning: Fine-tuning the model parameters for better accuracy.
- Deployment: Deploying the model as a web service for real-time battle outcome predictions.
