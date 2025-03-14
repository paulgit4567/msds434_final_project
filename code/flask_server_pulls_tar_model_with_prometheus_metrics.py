import os
import tarfile
import pandas as pd
import joblib
from flask import Flask, request, jsonify, Response
import boto3
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

s3_bucket = "paul-golf-model-and-data-bucket"
model_prefix = "model/"

# https://stackoverflow.com/questions/74556459/how-do-you-locally-load-model-tar-gz-file-from-sagemaker
# https://docs.aws.amazon.com/sagemaker/latest/dg/cdf-training.html
ec2_tar_path = "/app/model.tar.gz"
extract_dir = "/app"

s3 = boto3.client("s3")

# list everything in an s3
# https://www.youtube.com/watch?v=ZR6adef3fCM
# https://stackoverflow.com/questions/30249069/listing-contents-of-a-bucket-with-boto3
def get_latest_model_tar(bucket, prefix):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' not in response:
        raise Exception("No objects found under prefix")
    tar_objects = [obj for obj in response['Contents'] if obj['Key'].endswith('.tar.gz')]
    if not tar_objects:
        raise Exception("no model found")

    # pull most recent model file
    # https://www.radishlogic.com/aws/boto3/how-to-get-all-versions-of-a-single-object-file-in-an-aws-s3-bucket-using-python-boto3/

    latest_obj = max(tar_objects, key=lambda x: x['LastModified'])
    print(f"key: {latest_obj['Key']} (last modified on: {latest_obj['LastModified']})")
    return latest_obj['Key']

try:
    s3_model_tar_path = get_latest_model_tar(s3_bucket, model_prefix)
except Exception as e:
    print("error:", e)
    exit(1)

# pull model from s3
# https://stackoverflow.com/questions/42935034/python-boto-3-how-to-retrieve-download-files-from-aws-s3
# https://docs.aws.amazon.com/AmazonS3/latest/userguide/download-objects.html
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-example-download-file.html
try:
    s3.download_file(s3_bucket, s3_model_tar_path, ec2_tar_path)
    print(f"downloaded {s3_bucket}/{s3_model_tar_path} to {ec2_tar_path}")
except Exception as e:
    print("error:", e)
    exit(1)

# extract tar https://www.youtube.com/watch?v=k9T_7B74Kko
try:
    with tarfile.open(ec2_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
except Exception as e:
    print("error:", e)
    exit(1)

extracted_model_path = os.path.join(extract_dir, "sg_t2g_model_v2.pkl")

try:
    loaded = joblib.load(extracted_model_path)
    model = loaded["model"]
except Exception as e:
    print("error:", e)
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

        num_predictions.labels(endpoint="/predict").inc()

        return jsonify({"predicted sg_t2g": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
