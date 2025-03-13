import os
import tarfile
import pandas as pd
import joblib
from flask import Flask, request, jsonify, Response
import boto3
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# S3 bucket and prefix where model model files (.tar.gz) are stored
s3_bucket = "paul-golf-model-and-data-bucket"
model_prefix = "model/"

# Local paths for tarball and extraction directory
ec2_tar_path = "/app/model.tar.gz"
extract_dir = "/app"

s3 = boto3.client("s3")


def get_latest_model_tar(bucket, prefix):
    """List objects under the prefix and return the key of the most recently modified tarball."""
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' not in response:
        raise Exception("No objects found under prefix")

    # Filter only tar.gz files
    tar_objects = [obj for obj in response['Contents'] if obj['Key'].endswith('.tar.gz')]
    if not tar_objects:
        raise Exception("No tar.gz files found under prefix")

    # Pick the tarball with the most recent LastModified timestamp
    latest_obj = max(tar_objects, key=lambda x: x['LastModified'])
    print(f"Latest model tarball key: {latest_obj['Key']} (LastModified: {latest_obj['LastModified']})")
    return latest_obj['Key']


# Get the key for the most recent model tarball
try:
    s3_model_tar_path = get_latest_model_tar(s3_bucket, model_prefix)
except Exception as e:
    print("Error finding latest model tarball:", e)
    exit(1)

# Download the latest model tarball from S3
try:
    s3.download_file(s3_bucket, s3_model_tar_path, ec2_tar_path)
    print(f"Downloaded {s3_bucket}/{s3_model_tar_path} to {ec2_tar_path}")
except Exception as e:
    print("Model tar download error:", e)
    exit(1)

# Extract the tarball to the specified directory
try:
    with tarfile.open(ec2_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print("Model tar extracted successfully.")
except Exception as e:
    print("Error extracting model tarball:", e)
    exit(1)

extracted_model_path = os.path.join(extract_dir, "sg_t2g_model_v2.pkl")

try:
    model = joblib.load(extracted_model_path)
    print("Model loaded successfully from", extracted_model_path)
except Exception as e:
    print("Model load error:", e)
    exit(1)

app = Flask(__name__)

num_predictions = Counter('predictions_counter', 'Prediction requests', ['endpoint'])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        if not isinstance(input_data, list) or len(input_data) != 9:
            return jsonify({"error": "input error"}), 400

        expected_columns = [
            "sg_total", "driving_dist", "driving_acc", "gir", "scrambling",
            "prox_rgh", "prox_fw", "great_shots", "poor_shots"
        ]
        df = pd.DataFrame([input_data], columns=expected_columns)
        prediction = model.predict(df)

        # Increment the prediction counter
        num_predictions.labels(endpoint="/predict").inc()

        return jsonify({"predicted sg_t2g": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
