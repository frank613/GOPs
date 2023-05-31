# GOPs

## Pykaldi-scripts are the tools based on pykaldi and pure-kaldi are the scripts in C++ which follow kaldi-style implementation. 

## We use pykaldi for computing the GOP with gmm-based models, please find the scripts for frame/align versions of GOP under the foler Pykaldi-scripts

## Because we used kaldi-nnet2 for training the DNN-based models, and pykaldi only supports nnet3 for the time being, we can't directly use pykaldi which requires a convertion the nnet3. However, we use pure-kaldi here.

## All the methods require an alignment file as input. In the paper, we followed the standard mono-phone model recipe that trains the alignment model with librispeech train-clean-100.   
