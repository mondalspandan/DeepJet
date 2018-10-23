#!/usr/bin/env bash
### Job name
#BSUB -J Test-And
 
### File / path where STDOUT & STDERR will be written
###    %J is the job ID, %I is the array ID
#BSUB -o gpuTest-Cuda.%J.%I

#BSUB -gpu -
#BSUB -R pascal
 
### Request the time you need for execution in minutes
### Format [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 2:00
 
### Request vitual memory (in MB)
#BSUB -M 40000
 
### Request GPU
#BSUB -gpu -
#BSUB -R gpu
 
#module load cuda

export PATH="/home/ne020107/miniconda2/bin:$PATH"
INDIR=~/DDX/DeepJet/test
rm -r $INDIR
mkdir $INDIR
cd ~/DDX/DeepJet
source gpu_env.sh
cd Train
convertFromRoot.py -i ../train_list.txt -o $INDIR/dctrain -c TrainData_DeepDoubleC_lowest
python Train.py -i $INDIR/dctrain/dataCollection.dc -o $INDIR/training --batch 4096 --epochs 100 --resume
#cp -r $INDIR/training $INDIR/training_nodec
#python Train.py -i $INDIR/dctrain/dataCollection.dc -o $INDIR/training --batch 4096 --epochs 20 --resume --decor

#convertFromRoot.py -i ../list_multi_test.txt -o $INDIR/dctest --testdatafor $INDIR/training/trainsamples.dc
#rm -r $INDIR/res
#python BEval.py -i $INDIR/dctest/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training_nodec -o $INDIR/res
#rm -r $INDIR/res2
#python BEval.py -i $INDIR/dctest/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/res2

#convertFromRoot.py -i ../list_ddb_test_small.txt -o $INDIR/dctestddb --testdatafor $INDIR/training/trainsamples.dc
#python BEval.py -i $INDIR/dctestddb/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training_no_dec -o $INDIR/resddb
#python BEval.py -i $INDIR/dctestddb/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/res2ddb

