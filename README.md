# GOPs

## Pykaldi-scripts are the tools based on pykaldi and pure-kaldi are the scripts in C++ which follows kaldi-style implementation. 

## We use pykaldi for computing the GOP with the gmm-based models, please find the scripts for frame/align versions of GOP under the foler Pykaldi-scripts

## For the DNN based GOPs, because we used nnet2 for training, and pykaldi only supports nnet3 for the time being. One possible way is to convert the nnet2 to nnet3 and use pykaldi. However, we use pure-kaldi here.

## All the methods require an alignment file as input. In the paper, we followed the standard mono-phone model recipe that trains the model with librispeech train-clean-100.   
