#!/bin/bash
 
#SBATCH --partition=all
#SBATCH --constraint=P100
#SBATCH --time=60:00:00           # Maximum time request
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --workdir   /home/anovak/data/Mauro/DeepJet/run
#SBATCH --job-name  DeepJetBoostDDC
#SBATCH --output    run-%j.out  # File to which STDOUT will be written
#SBATCH --error     run-%j.out  # File to which STDERR will be written
#SBATCH --mail-type FAIL, END                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user novak@physik.rwth-aachen.de  # Email to which notifications will be sent

INDIR=~/data/boost/DDCfull100
#rm -r $INDIR
#mkdir $INDIR   
cd ~/data/Mauro/DeepJet
source gpu_env.sh
cd Train
#convertFromRoot.py -i ../Hcc_train_list.txt -o $INDIR/dctrain -c TrainData_deepDoubleC_lowest
##python Train.py -i $INDIR/dctrain/dataCollection.dc -o $INDIR/training --batch 2048 --epochs 100 --resume
convertFromRoot.py -i ../Hcc_test_list.txt -o $INDIR/dctestpre --testdatafor $INDIR/training/trainsamples.dc
convertFromRoot.py -i ../dev-multi-test.txt -o $INDIR/dctestpre-multi --testdatafor $INDIR/training/trainsamples.dc
python BEval.py -i $INDIR/dctestpre/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/evalpre
python BEval.py -i $INDIR/dctestpre-multi/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/eval-allpre

