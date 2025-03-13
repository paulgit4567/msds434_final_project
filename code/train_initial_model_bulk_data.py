import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

def train_and_save_model(csv_path, model_path):

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    feature_cols = ['sg_total', 'driving_dist', 'driving_acc', 'gir', 'scrambling',
                    'prox_rgh', 'prox_fw', 'great_shots', 'poor_shots']
    target_col = 'sg_t2g'
    X = df[feature_cols]
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SGDRegressor(loss="squared_error", max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump({'scaler': scaler, 'model': model}, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    csv_file = "/Users/paul/Documents/School/a. Northwestern_MSIS/f. Quarter 6/MSDS434 Cloud Computing/z_Final Project/data/simple_golf_stats_db only pga with sg cat/player_round_data_pga_tour_with_sg_data.csv"
    model_file = "/Users/paul/Documents/School/a. Northwestern_MSIS/f. Quarter 6/MSDS434 Cloud Computing/z_Final Project/model/sg_t2g_model_v2.pkl"
    train_and_save_model(csv_file, model_file)
