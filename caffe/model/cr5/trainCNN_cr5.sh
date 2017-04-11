mkdir Snapshots/cr_5_CNN_10172016 #Create a folder to save the caffe models (measurement rate is 0.04)
../../../../../build/tools/caffe train -solver videoNetCNN_cr5_solver.prototxt -gpu 2

