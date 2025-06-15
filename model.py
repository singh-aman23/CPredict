import pandas as pd
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import sklearn

def fetch_user_contests(handle):
    url = f"https://codeforces.com/api/user.rating?handle={handle}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        return pd.DataFrame(data['result'])
    else:
        raise Exception("Failed to fetch user data")

def train_and_save_model():
    df = fetch_user_contests('tourist')
    df['ratingChange'] = df['newRating'] - df['oldRating']
    df = df[['rank', 'oldRating', 'ratingChange']]
    X = df[['oldRating', 'rank']]
    y = df['ratingChange']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    save_path = os.path.abspath("rating_model.pkl")
    joblib.dump(rf_model, save_path)
    print(f"✅ Model saved at: {save_path} (scikit-learn {sklearn.__version__})")

if __name__ == "__main__":
    print("🧭 Current working directory:", os.getcwd())
    train_and_save_model()
