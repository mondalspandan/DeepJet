#!/bin/bash
 
#SBATCH --partition=all
#SBATCH --constraint="GPU"
##SBATCH --constraint="GPUx2"
#SBATCH --time=03:00:00           # Maximum time request
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --workdir   /home/anovak/data/Mauro/DeepJet/run
#SBATCH --job-name  bdt
#SBATCH --output    bdt-%j.out  # File to which STDOUT will be written
#SBATCH --error     bdt-%j.out  # File to which STDERR will be written
#SBATCH --mail-type FAIL, END                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user novak@physik.rwth-aachen.de  # Email to which notifications will be sent

INDIR=~/data/BDTfiles/run
rm -r $INDIR
mkdir $INDIR   
cd ~/data/Mauro/DeepJet
source gpu_env.sh
cd Train

convertFromRoot.py -i ../bdt_train.txt -o $INDIR/dctrain -c TrainData_VHbb_bdt

python BTrain.py -i $INDIR/dctrain/dataCollection.dc -o $INDIR/training  --batch 1024 --epochs 4

convertFromRoot.py -i ../bdt_test.txt -o $INDIR/dctest --testdatafor $INDIR/training/trainsamples.dc

python BEval.py -i $INDIR/dctest/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/eval

 
