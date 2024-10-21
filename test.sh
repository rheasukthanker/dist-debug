#!/bin/bash -l

#SBATCH --time=2:00:00   # walltime
#SBATCH --nodes=2
#SBATCH --account=projectnucleus
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4 # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:4
#SBATCH --mem=50GB   # memory per CPU core
#SBATCH -p develbooster
#SBATCH --exclusive


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10


export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export MASTER_PORT=12802
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR

module load Stages/2024
module load CUDA/12
module load GCC/12.3.0
module load Python/3.11.3
module purge
module load Stages/2024
module load GCC OpenMPI
# Some base modules commonly used in AI
module load mpi4py numba tqdm matplotlib IPython SciPy-Stack bokeh git
module load Flask Seaborn OpenCV

# ML Frameworks
module load  PyTorch scikit-learn torchvision PyTorch-Lightning
module load tensorboard
ml purge
ml Stages/2024
ml StdEnv
ml Python/3.11.3
ml CUDA/12
#ml cuDNN/8.9.5.29-CUDA-12
#ml NCCL/default-CUDA-12
#ml PyTorch/2.1.2
#ml torchvision
#ml tensorboard
#MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
# Allow communication over InfiniBand cells.
# adding ?~@~\i?~@~] to hostname is crucial, otherwise compute nodes will not be able to communicate
# export MASTER_ADDR="${MASTER_ADDR}i"
export MASTER_PORT=23456
#source /p/project/projectnucleus/sukthanker1/litgpt_env/bin/activate
export TRITON_CACHE_DIR="cache/"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
export PL_TORCH_DISTRIBUTED_BACKEND=nccl
# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent GLOO not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0
source /p/project/projectnucleus/sukthanker1/litgpt_env/bin/activate
export PYTHONPATH=.

srun --cpu_bind=v --accel-bind=gn python finetune/finetune.py EleutherAI/pythia-410m  --devices 4 --num_nodes 2 --train.learning_rate 2e-5  --train.lr_warmup_steps 300 --sampling_strategy "random" --out_dir test  --search_space hw_gpt_bench