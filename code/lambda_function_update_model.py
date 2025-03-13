# triggered every week after get_new_events
# pulls the most recent csv
# spins up sagemaker training instance
# fine tunes the model using fine_tune.py on the sg instance
# sg cleans data, fine tunes model then saves the fine tuned model to the model folder in s3

import boto3
import time

sagemaker = boto3.client('sagemaker')


def lambda_handler(event, context):
    # puls just the info for the most recent csv that triggered update_model to run
    # https://stackoverflow.com/questions/53891128/how-to-get-s3-bucket-name-and-key-of-a-file-from-an-event-in-lambda

    bucket_name = event['Records'][0]['s3']['bucket']['name']
    uploaded_file_key = event['Records'][0]['s3']['object']['key']

    # https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-docker-containers-scikit-learn-spark.html
    training_image = '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3'

    training_job_name = f"fine-tune-job-{int(time.time())}"

    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/create_training_job.html
    # https://www.youtube.com/watch?v=-iU36P8hizs

    # https://repost.aws/knowledge-center/lambda-sagemaker-create-notebook
    # https://www.youtube.com/watch?v=xe9-GZ1tX28
    # https://github.com/aws/sagemaker-training-toolkit/issues/82

    response = sagemaker.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification={
            'TrainingImage': training_image,
            'TrainingInputMode': 'File',
        },
        RoleArn='arn:aws:iam::051826733739:role/sagemaker_execution',
        # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html#your-algorithms-training-algo-running-container-inputdataconfig
        InputDataConfig=[
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3Uri': f's3://{bucket_name}/{uploaded_file_key}',
                        'S3DataType': 'S3Prefix',
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'ContentType': 'text/csv',
                'InputMode': 'File'
            },
            {
                # Channel for model file
                'ChannelName': 'model',
                'DataSource': {
                    'S3DataSource': {
                        'S3Uri': 's3://paul-golf-model-and-data-bucket/model',
                        'S3DataType': 'S3Prefix',
                        'S3DataDistributionType': 'FullyReplicated',
                    }
                },
                'ContentType': 'application/octet-stream',
                'InputMode': 'File'
            }
        ],
        OutputDataConfig={
            'S3OutputPath': 's3://paul-golf-model-and-data-bucket/model/'
        },
        ResourceConfig={
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 20
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 3600
        },
        HyperParameters={
            # imporintg training script
            # https://docs.aws.amazon.com/sagemaker/latest/dg/prebuilt-containers-extend.html
            'sagemaker_program': 'fine_tune.py',
            'sagemaker_submit_directory': 's3://paul-golf-model-and-data-bucket/scripts/fine_tune.tar.gz'
        }
    )

    print(f"traiing job {training_job_name} started.")
    return {
        'statusCode': 200,
        'body': f"job started: {training_job_name}"
    }
