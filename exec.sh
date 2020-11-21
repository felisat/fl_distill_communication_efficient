#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=felix.sattler@hhi.fraunhofer.de
#SBATCH --output=out/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1


cmdargs=$1

hyperparameters=' [{
	"dataset" : ["cifar10"], 
	"distill_dataset" : ["stl10"],
	"net" : ["resnet8_bn"],
	

	"n_clients" : [20],
	"classes_per_client" : [0.1, 100.0],
	"balancedness" : [1.0],


	"communication_rounds" : [20],
	"participation_rate" : [0.4],
	"local_epochs" : [5],
	"distill_iter" : [2000],
	"n_distill" : [51200],
	"reset_model" : [false],
	"co_distill" : [false],
	"co_distill_iter" : [2000], 

	"active" : [null],
	
	
	"batch_size" : [128],
	"aggregation_mode" : ["FDquantdown"],
	"quantization_bits" : [1, 2, 4, 8, 16, 32],
	"quantization_bits_down" : [1, 2, 4, 8, 16, 32],

	"save_softlabels" : [false],
	

	"pretrained" : ["simclr_resnet8_stl10_100epochs.pth"],

	"save_model" : [null],
	"log_frequency" : [-100],
	"log_path" : ["quantize_up_down_cifar/"],
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






