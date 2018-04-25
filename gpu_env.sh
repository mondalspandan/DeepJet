
export DEEPJETCORE=../DeepJetCore

THISDIR=`pwd`
cd $DEEPJETCORE
source gpu_env.sh
cd $THISDIR
export PYTHONPATH=`pwd`/modules:$PYTHONPATH
export DEEPJET=`pwd`

# Maxwell Specific
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/TensorFlow/lib64/

