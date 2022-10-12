export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'
export MLFLOW_TRACKING_URI='https://user-mbenxsalha-561205.user.lab.sspcloud.fr/'
export MLFLOW_EXPERIMENT_NAME="Default"

run_name="test"
MLFLOW_EXPERIMENT_NAME="Default"
# set the hyper parameters
lr="0.01"
weight_decay="0.001"
epochs="1"
batch_size="64"

#directory of MLProject file
path="/home/onyxia/work/"

mlflow run ${path} -P remote_server_uri=${MLFLOW_TRACKING_URI} \
-P experiment_name=${MLFLOW_EXPERIMENT_NAME} -P run_name=${run_name}
-P lr=${lr} -P weight_decay=${weight_decay} -P epochs=${epochs} -P batch_size=${batch_size}