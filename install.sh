#!/bin/bash

pip install -r requirements.txt
pip install .
source venv/bin/activate

ipcluster start -n 4 & 


pyqmri --config config --model VFA --data ../23052018_VFA_89.h5  --reg_type LOG --slices 1 --double_precision 1
#--streamed 0 --imagespace 1

#--useCGguess 0
ipcluster stop
