"""
A simple python MLP classifier for the mushroom dataset found here: https://www.kaggle.com/uciml/mushroom-classification
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('mushrooms.csv')
print(len(df.columns))

y_df = df["class"]
x_df = df[["cap-shape", "cap-surface", "cap-color", "bruises", "odor",
                 "gill-attachment", "gill-spacing", "gill-size", "gill-color",
                 "stalk-shape", "stalk-root", "stalk-surface-above-ring",
                 "stalk-surface-below-ring", "stalk-color-above-ring",
                 "stalk-color-below-ring", "veil-type", "veil-color",
                 "ring-number", "ring-type", "spore-print-color",
                 "population", "habitat"]]

"""
Converts the data frame of string values into unique numerical values 
for the x and y dataframes and converts them into numpy arrays
"""
label_encoder = preprocessing.LabelEncoder()
encoded_x_df = x_df.apply(preprocessing.LabelEncoder().fit_transform)
x = np.array(encoded_x_df)

label_encoder.fit(y_df)
y = label_encoder.transform(y_df)
y = np.array(y)

# Test train and split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

clf = MLPClassifier(solver="adam",
                    batch_size=50,
                    activation="relu",
                    hidden_layer_sizes=(100, 50, 25),
                    random_state=42, max_iter=2000)

clf.fit(x_train, y_train)

"""
Validating the model results
if score is 1.0 then the model is 100% accurate
"""
print(f"Score: {clf.score(x_val, y_val)}")

