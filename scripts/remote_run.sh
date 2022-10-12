#set environment variables
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'
export MLFLOW_TRACKING_URI='https://user-mbenxsalha-561205.user.lab.sspcloud.fr/'
MLFLOW_EXPERIMENT_NAME="Default"
export MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME}

mlflow_run_name="remote"
# set the hyper parameters
lr="0.01"
weight_decay="0.001"
epochs="1"
batch_size="64"

mlflow run https://github.com/amine-bs/mlflow-argo.git -P remote_server_uri=${MLFLOW_TRACKING_URI} \
-P mlflow_experiment_name=${MLFLOW_EXPERIMENT_NAME} -P mlflow_run_name=${mlflow_run_name}
-P lr=${lr} -P weight_decay=${weight_decay} -P epochs=${epochs} -P batch_size=${batch_size}