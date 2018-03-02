
import os
from argparse import ArgumentParser

path = '/eos/user/a/anovak/public/merged_QCDHccHbb/train/'

parser = ArgumentParser()
parser.add_argument("--train", help="Path to training sample", default=path)
parser.add_argument("--test", help="Path to testing sample", default=path.replace("train", "test") )
args=parser.parse_args()


### Write train input list
path = args.train
if path.endswith("/"): path = path[:-1]
f = open('train_list.txt', 'w')
for i in os.listdir(path):
	if not i.endswith("root"): continue
	f.write(path+"/"+i+'\n')
f.close()


#path = '/eos/user/a/anovak/test2files/test'
#path = '/eos/user/a/anovak/train_files'

### Write test input list
path = args.test
if path.endswith("/"): path = path[:-1]
f = open('test_list.txt', 'w')
for i in os.listdir(path):
	if not i.endswith("root"): continue
	f.write(path+"/"+i+'\n')
f.close()
