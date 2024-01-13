#!/bin/bash
#SBATCH --partition=serial
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --job-name=cceph-fci
#SBATCH --mem=0

module purge
module load gcc/9.2.0
module load binutils/2.26
module load cmake-3.6.2 

export OMP_NUM_THREADS=28;
export MKL_NUM_THREADS=28;
export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE;
echo $PYSCF_MAX_MEMORY

source /home/yangjunjie/intel/oneapi/setvars.sh --force;
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
conda activate py310-block2

export TMPDIR=/scratch/global/yangjunjie/$SLURM_JOB_NAME/$SLURM_JOB_ID/
export PYSCF_TMPDIR=$TMPDIR
echo TMPDIR       = $TMPDIR
echo PYSCF_TMPDIR = $PYSCF_TMPDIR
mkdir -p $TMPDIR

export PYTHONPATH=/home/yangjunjie/work/cc-eph/cceph-main
export PYTHONPATH=/home/yangjunjie/work/cc-eph/epcc-hol:$PYTHONPATH
export PYTHONPATH=/home/yangjunjie/work/cc-eph/wick-dev:$PYTHONPATH
export PYTHONPATH=/home/yangjunjie/work/cc-eph/cqcpy-master:$PYTHONPATH
export PYTHONPATH=/home/yangjunjie/packages/pyscf/pyscf-main/:$PYTHONPATH
export PYTHONPATH=/home/yangjunjie/work/holstein-results/src/:$PYTHONPATH
export PYTHONUNBUFFERED=TRUE;
