#set environment variables
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'
export MLFLOW_TRACKING_URI='https://user-mbenxsalha-84763.user.lab.sspcloud.fr/'
export MLFLOW_EXPERIMENT_NAME="Default"


# set hyper parameters
lr="0.01"
weight_decay="0.001"
epochs="1"
batch_size="64"
mlflow_experiment_name="Default"
mlflow_run_name="test"

root_path="/home/onyxia/work/"

python ${root_path}/main.py --lr lr --weight_decay weight_decay --epochs epochs --batch_size batch_size --mlflow_experiment_name mlflow_experiment_name --mlflow_run_name mlflow_run_name