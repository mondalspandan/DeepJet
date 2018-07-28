#!/bin/bash
 
#SBATCH --partition=all
#SBATCH --constraint=P100
#SBATCH --no-requeue 
#SBATCH --time=60:00:00           # Maximum time request
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --workdir   /home/anovak/data/Mauro/DeepJet/run
#SBATCH --job-name  DeepJetBoostDDC
#SBATCH --output    run-%j.out  # File to which STDOUT will be written
#SBATCH --error     run-%j.out  # File to which STDERR will be written
#SBATCH --mail-type FAIL, END                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user novak@physik.rwth-aachen.de  # Email to which notifications will be sent

INDIR=~/data/dev/DDB
#rm -r $INDIR
#mkdir $INDIR   
cd ~/data/Mauro/DeepJet
source gpu_env.sh
cd Train
#convertFromRoot.py -i ../list_ddb_train_small.txt -o $INDIR/dctrain -c TrainData_deepDoubleB_lowest
#cp -r $INDIR/training $INDIR/training_nodec
#convertFromRoot.py -i ../list_ddb_train_small.txt -o $INDIR/dctraindec -c TrainData_deepDoubleB_lowest
#python Train.py -i $INDIR/dctrain/dataCollection.dc -o $INDIR/training --batch 4096 --epochs 100 --resume
python Train.py -i $INDIR/dctraindec/dataCollection.dc -o $INDIR/training --batch 4096 --epochs 3 --resume #--decor
#convertFromRoot.py -i ../list_ddb_test_small.txt -o $INDIR/dctest --testdatafor $INDIR/training/trainsamples.dc
#python BEval.py -i $INDIR/dctest/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/res
#python BEval.py -i $INDIR/dctest/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/res2

