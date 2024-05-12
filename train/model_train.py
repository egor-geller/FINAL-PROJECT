import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("cleaned.csv")

X = df.drop(columns=['Heart_Disease', 'Unnamed: 0'])
y = df['Heart_Disease']
test_size = .25
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 4, 5]
}

best_model = GradientBoostingClassifier(learning_rate=0.01, max_depth=3, n_estimators=100)
best_model.fit(X_train, y_train)

with open("/usr/src/app/train/best_model.pkl", "wb") as f:
    pickle.dump({
        'model': best_model
        }, f)
