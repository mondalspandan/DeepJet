
export DEEPJETCORE=../DeepJetCore

THISDIR=`pwd`
cd $DEEPJETCORE
source gpu_env.sh
cd $THISDIR
export PYTHONPATH=`pwd`/modules:$PYTHONPATH
export DEEPJET=`pwd`

# for caltech cluster:
#rm -rf $HOME/.nv
#export CUDA_CACHE_PATH=/tmp/$USER/cuda/
#mkdir -p $CUDA_CACHE_PATH

# for Maxwell (DESY) 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/TensorFlow/lib64/

