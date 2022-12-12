
#!/bin/bash

#SBATCH --job-name=ML_Extreme
#SBATCH --time=1-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4571

MY_NUM_THREADS=$SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=$MY_NUM_THREADS


module purge

module load GCC/11.2.0 
module load OpenMPI/4.1.1
module load SciPy-bundle/2021.10

source ~/env/bin/activate

srun python3 your_code.py

deactivate
