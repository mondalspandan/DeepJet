

DeepJet: Repository for training and evaluation of deep neural networks for Jet identification
===============================================================================

This package depends on DeepJetCore - original at (https://github.com/DL4Jets/DeepJetCore)
For Maxwell Cluster in DESY and working with RWTH get https://github.com/anovak10/DeepJetCore and follow the setup

Setup
==========

The DeepJet package and DeepJetCore have to share the same parent directory

Usage (Maxwell)
==============

Connect to desy and maxwell
```
ssh max-wgs
```

To run interactively:
```
salloc -N 1 --partition=all --constraint=GPU --time=<minutes>
```

Alternatively add the following to your .bashrc to get allocation for x hours with ``` getgpu x ```
```
getgpu () {
   salloc -N 1 --partition=all --constraint=GPU --time=$((60 * $1))
}
```

ssh to the machine that was allocated to you. For example
```
ssh max-wng001
```
```
cd <your working dir>/DeepJet
source gpu_env.sh
```


The preparation for the training consists of the following steps
====

- define the data structure for the training (example in modules/datastructures/TrainData_template.py)
  for simplicity, copy the file to TrainData_template.py and adjust it. 
  Define a new class name (e.g. TrainData_template), leave the inheritance untouched

- convert the root file to the data strucure for training using DeepJetCore tools:
```
convertFromRoot.py -i /path/to/the/root/ntuple/list_of_root_files.txt -o /output/path/that/needs/some/disk/space -c TrainData_myclass
```
- You can use the following script to create the lists (if you store the files in a train and test directory within one parent you can only specify test
```
  python list_writer.py --train <path/to/directory/of/files/train> --test <path/to/directory/of/files/test>
``` 
Example use:
```
mkdir run
INDIR=run # Make a variable for a parent directory
convertFromRoot.py -i train_list.txt -o $INDIR/dctrain -c TrainData_deepDoubleC_db_cpf_sv_reduced
```
Training
====

Run the training (for now BTrain for beta)
```
cd Train
python BTrain.py -i $INDIR/dctrain/dataCollection.dc -o $INDIR/training  --batch 1024 --epochs 50

```
Evaluation
====

After the training has finished, the performance can be evaluated.
The evaluation consists of a few steps:

1) converting the test data
```
cd ..
convertFromRoot.py -i test_list.txt -o $INDIR/dctest --testdatafor $INDIR/training/trainsamples.dc
```

2) Evaluate

```
cd Train
python Eval.py -i $INDIR/dctest/dataCollection.dc -t $INDIR/dctrain/dataCollection.dc -d $INDIR/training -o $INDIR/eval
```

Output .pkl file and some plots will be stored in $INDIR/eval

To use Maxwell Batch
====
Example config file can be found in run/
```
# To run binary Hcc vs QCD training
sbatch run/baseDDC.sh

# To run binary Hcc vs Hbb training
sbatch run/baseDDCvB.sh

# To run multiclassifier for Hcc, Hbb, QCD (gcc, gbb, Light)
sbatch run/basemulti.sh

# To see job output updated in real time
tail -f run/run-<jobid>.out 
# To show que
squeue -u username 
# To cancel a job 
scancel jobid # To cancel job
```


Currently unused:
====
2) applying the trained model to the test data

```
predict.py <output dir of training>/KERAS_model.h5  /output/path/for/test/data/dataCollection.dc <output directory>
```
This creates output trees. and a tree_association.txt file that is input to the plotting tools

There is a set of plotting tools with examples in 
DeepJet/Train/Plotting


