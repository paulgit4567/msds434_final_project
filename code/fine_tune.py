

# used by sagemaker instance to update the model and save a new updated model back to the s3 model folder
# called by the update_model lambda function to spin up a sagemaker instance to fine tune the model saved in the s3
# update_model runs a few hours after get_new_events each week


import os
import numpy as np
import pandas as pd
import joblib
import time
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# print('test')

def update_model():
    # print('run update model function')

    # SM_CHANNEL_MODEL is the training data that update_model points to here
    # 'ChannelName': 'training',
    # 'DataSource': {
    #     'S3DataSource': {
    #         'S3Uri': f's3://{bucket_name}/{uploaded_file_key}',

    # https://discuss.huggingface.co/t/incrementally-finetuning-a-hf-model-in-sagemaker/17443/6
    # load the sagemaker model info
    model_dir = os.environ['SM_CHANNEL_MODEL']
    data_dir = os.environ['SM_CHANNEL_TRAINING']

    model_scaler_path = os.path.join(model_dir, 'sg_t2g_model_v2.pkl')
    saved = joblib.load(model_scaler_path)
    model = saved["model"]
    scaler = saved["scaler"]

    # pulls csvs that have been uploaded in the past 5 days
    # ensures that only new data is used to fine tune the model
    cutoff_timestamp = time.time() - (5 * 24 * 60 * 60)
    training_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            full_path = os.path.join(data_dir, file)
            #print(full_path)
            file_mod_time = os.path.getmtime(full_path)
            #print(file_mod_time)

            if file_mod_time > cutoff_timestamp:
                training_files.append(full_path)
    feature_cols = ['sg_total', 'driving_dist', 'driving_acc', 'gir', 'scrambling',
                    'prox_rgh', 'prox_fw', 'great_shots', 'poor_shots']
    target_col = 'sg_t2g'

    # cycles through csvs
    # cleans nulls, creates feature and label columns, and fine tunes the model
    for file_path in training_files:
        #print(f'model {file_path}')
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        #print(df.head())
        if df.empty:
            continue
        X = df[feature_cols]
        y = df[target_col]
        X_scaled = scaler.transform(X)
        model.partial_fit(X_scaled, y)

    output_path = os.path.join(os.environ['SM_MODEL_DIR'], 'sg_t2g_model_v2.pkl')
    joblib.dump({"model": model, "scaler": scaler}, output_path)
    print("new model saved")

if __name__ == '__main__':
    update_model()
