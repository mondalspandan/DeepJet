#!/bin/bash
 
#SBATCH --partition=all
#SBATCH --constraint=P100
#SBATCH --no-requeue 
#SBATCH --time=40:00:00           # Maximum time request
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --workdir   /home/spmondal/DNN_test/DeepJet_94xadv/run
#SBATCH --job-name  DDCvBvL_94X_50p
#SBATCH --output    run-%j.out  # File to which STDOUT will be written
#SBATCH --error     run-%j.out  # File to which STDERR will be written
#SBATCH --mail-type FAIL, END                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user mondal@physik.rwth-aachen.de  # Email to which notifications will be sent

INDIR=/beegfs/desy/user/spmondal/DNN/181103_adv_94x50p
rm -r $INDIR
mkdir $INDIR   
cd /home/spmondal/DNN_test/DeepJet_94xadv
source gpu_env.sh
cd Train
convertFromRoot.py -i ../train_list_94x_50p.txt -o $INDIR/dctrain -c TrainData_DeepDoubleX_3lab
#python Train.py -i $INDIR/dctrain/dataCollection.dc -o $INDIR/training --batch 4096 --epochs 2 --resume 
#cp -r $INDIR/training $INDIR/training_nodec
python Train.py -i $INDIR/dctrain/dataCollection.dc -o $INDIR/training --batch 4096 --epochs 100 --resume --decor --loss loss_reg --lambda-adv 1 --classes 3

convertFromRoot.py -i ../test_list_94x_50p.txt -o $INDIR/dctest --testdatafor $INDIR/training/trainsamples.dc
#python Eval.py -i $INDIR/dctest/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training_nodec -o $INDIR/res
python Eval.py -i $INDIR/dctest/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/res_adv --adv --era 2017


