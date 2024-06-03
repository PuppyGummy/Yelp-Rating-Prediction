import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the json file
df = pd.read_json('yelp_academic_dataset_review.json', lines=True)

# Drop the columns that are not needed
df = df.drop(['review_id', 'user_id', 'business_id', 'date'], axis=1)

X = df['text']
y1 = df['stars']
y2 = df['useful']
y3 = df['funny']
y4 = df['cool']