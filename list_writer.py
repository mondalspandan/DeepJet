
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--train", help="Path to training sample")
parser.add_argument("--test", help="Path to testing sample", default=None)
parser.add_argument("--listname", help="'(train/test)_'+listname+'.root'", default="list")
args=parser.parse_args()

if args.test == None:
	args.test = args.train.replace("train", "test")


### Write train input list
path = args.train
if path.endswith("/"): path = path[:-1]
f = open('train_'+args.listname+'.txt', 'w')
for i in os.listdir(path):
	if not i.endswith("root"): continue
	f.write(path+"/"+i+'\n')
f.close()


#path = '/eos/user/a/anovak/test2files/test'
#path = '/eos/user/a/anovak/train_files'

### Write test input list
path = args.test
if path.endswith("/"): path = path[:-1]
f = open('test_'+args.listname+'.txt', 'w')
for i in os.listdir(path):
	if not i.endswith("root"): continue
	f.write(path+"/"+i+'\n')
f.close()
