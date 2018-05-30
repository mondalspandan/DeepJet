#!/bin/bash
 
#SBATCH --partition=all
#SBATCH --constraint=P100
#SBATCH --time=30:00:00           # Maximum time request
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --workdir   /home/anovak/data/Mauro/DeepJet/run
#SBATCH --job-name  DeepJet
#SBATCH --output    run-%j.out  # File to which STDOUT will be written
#SBATCH --error     run-%j.out  # File to which STDERR will be written
#SBATCH --mail-type FAIL, END                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user novak@physik.rwth-aachen.de  # Email to which notifications will be sent

INDIR=~/data/base2/runmulti
rm -r $INDIR
mkdir $INDIR   
cd ~/data/Mauro/DeepJet
source gpu_env.sh
cd Train

convertFromRoot.py -i ../multi_train_list.txt -o $INDIR/dctrain -c TrainData_deepDoubleC_db_cpf_sv_reduced

python Train.py -i $INDIR/dctrain/dataCollection.dc -o $INDIR/training --batch 1024 --epochs 50

#rm -r $INDIR/dctest-multi
#rm -r $INDIR/dctes
#convertFromRoot.py -i ../Hcc_test_list.txt   -o $INDIR/dctest       --testdatafor $INDIR/training/trainsamples.dc
convertFromRoot.py -i ../multi_test_list.txt -o $INDIR/dctest-multi --testdatafor $INDIR/training/trainsamples.dc

#rm -r $INDIR/eval-all 
#python BEval.py -i $INDIR/dctest/dataCollection.dc       -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/eval
python BEval.py -i $INDIR/dctest-multi/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/eval-all

 
