#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=felix.sattler@hhi.fraunhofer.de
#SBATCH --output=out/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1


cmdargs=$1

hyperparameters=' [{
	"dataset" : ["mnist"], 
	"distill_dataset" : ["emnist"],
	"net" : ["lenet_mnist"],
	

	"n_clients" : [20],
	"classes_per_client" : [0.01],
	"balancedness" : [1.0],


	"communication_rounds" : [50],
	"participation_rate" : [0.4],
	"local_epochs" : [10],
	"distill_epochs" : [3],
	"n_distill" : [50000],
	"fallback" : [true],
	"lambda_outlier" : [1.0],
	"lambda_fedprox" : [0.0],
	"only_train_final_outlier_layer" : [false],


	"warmup_type": ["constant"],
	"mixture_coefficients" : [{"base":0.5, "public":0.5}],
	"distill_weight": [1],
	
	"batch_size" : [128],
	"aggregation_mode" : ["FAD+S"],
	

	"pretrained" : ["simclr_resnet8_stl10_100epochs.pth"],

	"save_model" : [null],
	"log_frequency" : [-100],
	"log_path" : ["train_only_final_outlier_layer/"],
	"job_id" : [['$SLURM_JOB_ID']]}]'



if [[ "$HOSTNAME" == *"vca"* ]]; then # Cluster

	RESULTS_PATH="/opt/small_files/"
	DATA_PATH="/opt/in_ram_data/"
	CHECKPOINT_PATH="/opt/checkpoints/"

	echo $hyperparameters
	source "/etc/slurm/local_job_dir.sh"

	export SINGULARITY_BINDPATH="$LOCAL_DATA:/data,$LOCAL_JOB_DIR:/mnt/output,./code:/opt/code,./checkpoints:/opt/checkpoints,./results:/opt/small_files,$HOME/in_ram_data:/opt/in_ram_data"
	singularity exec --nv $HOME/base_images/pytorch15.sif python -u /opt/code/federated_learning.py --hp="$hyperparameters" --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs

	mkdir -p results
	cp -r ${LOCAL_JOB_DIR}/. ${SLURM_SUBMIT_DIR}/results	


else # Local

	RESULTS_PATH="results/"
	DATA_PATH="/home/sattler/Data/PyTorch/"
	CHECKPOINT_PATH="checkpoints/"

	python -u code/federated_learning.py --hp="$hyperparameters" --RESULTS_PATH="$RESULTS_PATH" --DATA_PATH="$DATA_PATH" --CHECKPOINT_PATH="$CHECKPOINT_PATH" $cmdargs




fi






