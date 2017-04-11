# CSVideoNet
This is the implementation of the paper "CSVideoNet: A Recurrent Convolutional Neural Network for Compressive Sensing Video Reconstruction"(https://arxiv.org/abs/1612.05203).

1. 0.25_196_ExtractFrame.sh is used to extract frames from videos;
2. genPhi is used to generate sensing matrix;
3. generateTrainCNN.m is used to generate image blocks for pre-training key CNN;
4. "caffe/model/crX" directory contains all the necessary file for training key CNN. To start the training, run "sh trainCNN_crX.sh";
5. GenerateTrainData1ChanVarMix.m is used to generate image blocks for training the whole network;
6. extractFeatures_5_25.m is used to extract the intermediate features extracted by the key CNN, and the extracted features are used for training the whole framework;
7. "model/crX" directory contains all the files for training the whole framework. The pre-trained key CNN is loaded, further trained with the whole framework using the intermediate feature as input;
To start the training the whole framework, run "th trainVideoNet.lua".
