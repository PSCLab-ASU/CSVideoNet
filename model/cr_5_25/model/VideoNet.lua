require 'torch'
--require 'nn'
require 'rnn'
require 'loadcaffe'

local utils = require 'util.utils'

local VideoNet, parent = torch.class('nn.VideoNet', 'nn.Module')

function convRelu(model, inputLayers, hiddenLayers, cnnKernel, cnnStride, cnnPad)
  model:add(nn.SpatialConvolution(inputLayers, hiddenLayers, cnnKernel, cnnKernel, cnnStride, cnnStride, cnnPad, cnnPad))
  model:add(nn.SpatialBatchNormalization(hiddenLayers))
  model:add(nn.ReLU())
end

function convLayer(model, inputLayers, hiddenLayers, cnnKernel, cnnStride, cnnPad)
  model:add(nn.SpatialConvolution(inputLayers, hiddenLayers, cnnKernel, cnnKernel, cnnStride, cnnStride, cnnPad, cnnPad))
end

prototxt1 = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr5/videoNetCNN_cr5_deploy_10172016.prototxt'
binary1 = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr5/Snapshots/cr_5_CNN_10172016/videoNetCNN_5_iter_175000.caffemodel'

prototxt2 = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr25/videoNetCNN_cr25_model_deploy.prototxt'
binary2 = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr25/Snapshots/cr_25_CNN_10172016/videoNetCNN_25_iter_170000.caffemodel'

prototxt3 = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr50/videoNetCNN_cr50_model_deploy.prototxt'
binary3 = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr50/Snapshots/cr_50_CNN_10172016/videoNetCNN_50_iter_335000.caffemodel'

prototxt4 = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr100/videoNetCNN_cr100_model_deploy.prototxt'
binary4 = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr100/Snapshots/cr_100_CNN_10172016/videoNetCNN_100_iter_315000.caffemodel'

prototxt5 = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr1/videoNetCNN_cr1_deploy_10172016.prototxt'
binary5 = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr1/Snapshots/cr_1_CNN_10172016/videoNetCNN_5_iter_350000.caffemodel'

function VideoNet:__init(args)
  assert(args ~= nil)

  local batchNorm = utils.getArgs(args, 'batchNorm')
  local batchSize = utils.getArgs(args, 'batchSize')
  local dropout = utils.getArgs(args, 'dropout')
  local Height = utils.getArgs(args, 'Height')
  local Width = utils.getArgs(args, 'Width')
  local seqLength = utils.getArgs(args, 'seqLength')
  local numChannels = utils.getArgs(args, 'numChannels')
  local measurements1 = utils.getArgs(args, 'measurements1')
  local measurements2 = utils.getArgs(args, 'measurements2')
  local backend = utils.getArgs(args, 'backend')
  local cr = utils.getArgs(args, 'cr')
  local rnnModel = utils.getArgs(args, 'rnnModel')

  local cnn = {}
  cnn.numHidden = {}
  cnn.numHidden[1] = 64
  cnn.numHidden[2] = 32
  cnn.numHidden[3] = 1
  cnn.kernel = 3
  cnn.pad = 1
  cnn.stride = 1

  local lstm = {}
  lstm.input = cnn.numHidden[3] * Height * Width
  lstm.numHidden = {}
  lstm.numHidden[1] = 6 * Height * Width
  lstm.numHidden[2] = 6 * Height * Width
  lstm.numHidden[3] = numChannels * Height * Width
  lstm.numHidden[4] = 5 * Height * Width
  lstm.numHidden[5] = numChannels * Height * Width

    -- load CNN_5 and CNN_25
  if cr == 5 then
    modelCNN = loadcaffe.load(prototxt1, binary1, backend)
  elseif cr == 25 then
    modelCNN = loadcaffe.load(prototxt2, binary2, backend)
  elseif cr == 50 then
    modelCNN = loadcaffe.load(prototxt3, binary3, backend)
  elseif cr == 100 then
    modelCNN = loadcaffe.load(prototxt4, binary4, backend)
  elseif cr == 1 then
    modelCNN = loadcaffe.load(prototxt5, binary5, backend)
  end

  local start = 0
  modelCNN1 = nn.Sequential()
  for i = 1, #modelCNN do
    local layer = modelCNN:get(i)
    local layer_name = layer.name
    if layer_name == 'conv7' then
      start = 1
    end
    if start == 1 then
      modelCNN1:add(layer)
    end
  end

  measurements = measurements1 + measurements2

  parallel_model = nn.ParallelTable()  -- model that concatenates net1 and net2
  parallel_model:add(modelCNN1)

  modelCNN2 = nn.Sequential()
  modelCNN2:add(nn.View(-1, measurements))
  modelCNN2:add(nn.Linear(measurements, Height*Width))
  modelCNN2:add(nn.View(-1, Height, Width))
  modelCNN2:add(nn.View(-1, 1, Height, Width))
  convRelu(modelCNN2, numChannels, cnn.numHidden[1], cnn.kernel, cnn.stride, cnn.pad)
  convRelu(modelCNN2, cnn.numHidden[1], cnn.numHidden[2], cnn.kernel, cnn.stride, cnn.pad)
  convLayer(modelCNN2, cnn.numHidden[2], cnn.numHidden[3], cnn.kernel, cnn.stride, cnn.pad)
  modelCNN2:add(nn.View(-1, 1, cnn.numHidden[3], Height, Width))
  parallel_model:add(modelCNN2)

  modelCNN3 = nn.Sequential()
  modelCNN3:add(nn.View(-1, measurements))
  modelCNN3:add(nn.Linear(measurements, Height*Width))
  modelCNN3:add(nn.View(-1, Height, Width))
  modelCNN3:add(nn.View(-1, 1, Height, Width))
  convRelu(modelCNN3, numChannels, cnn.numHidden[1], cnn.kernel, cnn.stride,cnn.pad)
  convRelu(modelCNN3, cnn.numHidden[1], cnn.numHidden[2], cnn.kernel,cnn.stride, cnn.pad)
  convLayer(modelCNN3, cnn.numHidden[2], cnn.numHidden[3], cnn.kernel,cnn.stride, cnn.pad)
  modelCNN3:add(nn.View(-1, 1, cnn.numHidden[3], Height, Width))
  parallel_model:add(modelCNN3)

  modelCNN4 = nn.Sequential()
  modelCNN4:add(nn.View(-1, measurements))
  modelCNN4:add(nn.Linear(measurements, Height*Width))
  modelCNN4:add(nn.View(-1, Height, Width))
  modelCNN4:add(nn.View(-1, 1, Height, Width))
  convRelu(modelCNN4, numChannels, cnn.numHidden[1], cnn.kernel, cnn.stride,cnn.pad)
  convRelu(modelCNN4, cnn.numHidden[1], cnn.numHidden[2], cnn.kernel,cnn.stride, cnn.pad)
  convLayer(modelCNN4, cnn.numHidden[2], cnn.numHidden[3], cnn.kernel,cnn.stride, cnn.pad)
  modelCNN4:add(nn.View(-1, 1, cnn.numHidden[3], Height, Width))
  parallel_model:add(modelCNN4)

  modelCNN5 = nn.Sequential()
  modelCNN5:add(nn.View(-1, measurements))
  modelCNN5:add(nn.Linear(measurements, Height*Width))
  modelCNN5:add(nn.View(-1, Height, Width))
  modelCNN5:add(nn.View(-1, 1, Height, Width))
  convRelu(modelCNN5, numChannels, cnn.numHidden[1], cnn.kernel, cnn.stride,cnn.pad)
  convRelu(modelCNN5, cnn.numHidden[1], cnn.numHidden[2], cnn.kernel,cnn.stride, cnn.pad)
  convLayer(modelCNN5, cnn.numHidden[2], cnn.numHidden[3], cnn.kernel,cnn.stride, cnn.pad)
  modelCNN5:add(nn.View(-1, 1, cnn.numHidden[3], Height, Width))
  parallel_model:add(modelCNN5)

  modelCNN6 = nn.Sequential()
  modelCNN6:add(nn.View(-1, measurements))
  modelCNN6:add(nn.Linear(measurements, Height*Width))
  modelCNN6:add(nn.View(-1, Height, Width))
  modelCNN6:add(nn.View(-1, 1, Height, Width))
  convRelu(modelCNN6, numChannels, cnn.numHidden[1], cnn.kernel, cnn.stride,cnn.pad)
  convRelu(modelCNN6, cnn.numHidden[1], cnn.numHidden[2], cnn.kernel,cnn.stride, cnn.pad)
  convLayer(modelCNN6, cnn.numHidden[2], cnn.numHidden[3], cnn.kernel,cnn.stride, cnn.pad)
  modelCNN6:add(nn.View(-1, 1, cnn.numHidden[3], Height, Width))
  parallel_model:add(modelCNN6)

  modelCNN7 = nn.Sequential()
  modelCNN7:add(nn.View(-1, measurements))
  modelCNN7:add(nn.Linear(measurements, Height*Width))
  modelCNN7:add(nn.View(-1, Height, Width))
  modelCNN7:add(nn.View(-1, 1, Height, Width))
  convRelu(modelCNN7, numChannels, cnn.numHidden[1], cnn.kernel, cnn.stride,cnn.pad)
  convRelu(modelCNN7, cnn.numHidden[1], cnn.numHidden[2], cnn.kernel,cnn.stride, cnn.pad)
  convLayer(modelCNN7, cnn.numHidden[2], cnn.numHidden[3], cnn.kernel,cnn.stride, cnn.pad)
  modelCNN7:add(nn.View(-1, 1, cnn.numHidden[3], Height, Width))
  parallel_model:add(modelCNN7)

  modelCNN8 = nn.Sequential()
  modelCNN8:add(nn.View(-1, measurements))
  modelCNN8:add(nn.Linear(measurements, Height*Width))
  modelCNN8:add(nn.View(-1, Height, Width))
  modelCNN8:add(nn.View(-1, 1, Height, Width))
  convRelu(modelCNN8, numChannels, cnn.numHidden[1], cnn.kernel, cnn.stride,cnn.pad)
  convRelu(modelCNN8, cnn.numHidden[1], cnn.numHidden[2], cnn.kernel,cnn.stride, cnn.pad)
  convLayer(modelCNN8, cnn.numHidden[2], cnn.numHidden[3], cnn.kernel,cnn.stride, cnn.pad)
  modelCNN8:add(nn.View(-1, 1, cnn.numHidden[3], Height, Width))
  parallel_model:add(modelCNN8)

  modelCNN9 = nn.Sequential()
  modelCNN9:add(nn.View(-1, measurements))
  modelCNN9:add(nn.Linear(measurements, Height*Width))
  modelCNN9:add(nn.View(-1, Height, Width))
  modelCNN9:add(nn.View(-1, 1, Height, Width))
  convRelu(modelCNN9, numChannels, cnn.numHidden[1], cnn.kernel, cnn.stride,cnn.pad)
  convRelu(modelCNN9, cnn.numHidden[1], cnn.numHidden[2], cnn.kernel,cnn.stride, cnn.pad)
  convLayer(modelCNN9, cnn.numHidden[2], cnn.numHidden[3], cnn.kernel,cnn.stride, cnn.pad)
  modelCNN9:add(nn.View(-1, 1, cnn.numHidden[3], Height, Width))
  parallel_model:add(modelCNN9)

  modelCNN10 = nn.Sequential()
  modelCNN10:add(nn.View(-1, measurements))
  modelCNN10:add(nn.Linear(measurements, Height*Width))
  modelCNN10:add(nn.View(-1, Height, Width))
  modelCNN10:add(nn.View(-1, 1, Height, Width))
  convRelu(modelCNN10, numChannels, cnn.numHidden[1], cnn.kernel, cnn.stride,cnn.pad)
  convRelu(modelCNN10, cnn.numHidden[1], cnn.numHidden[2], cnn.kernel,cnn.stride, cnn.pad)
  convLayer(modelCNN10, cnn.numHidden[2], cnn.numHidden[3], cnn.kernel,cnn.stride, cnn.pad)
  modelCNN10:add(nn.View(-1, 1, cnn.numHidden[3], Height, Width))
  parallel_model:add(modelCNN10)

  self.model = nn.Sequential()
  self.model:add(parallel_model)
  self.model:add(nn.JoinTable(2))

  self.model:add(nn.View(batchSize, seqLength, cnn.numHidden[3]*Height*Width))-- [20, 16x32x32]
  self.model:add(nn.Transpose({1,2}))

  if rnnModel == 'LSTM' then
    rnn1 = nn.SeqLSTM(lstm.input, lstm.numHidden[1])
    rnn2 = nn.SeqLSTM(lstm.numHidden[1], lstm.numHidden[2])
    rnn3 = nn.SeqLSTM(lstm.numHidden[2], lstm.numHidden[3])
    rnn4 = nn.SeqLSTM(lstm.numHidden[3], lstm.numHidden[4])
    rnn5 = nn.SeqLSTM(lstm.numHidden[4], lstm.numHidden[5])
  elseif rnnModel == 'GRU' then
    rnn1 = nn.seqGRU(lstm.input, lstm.numHidden[1])
    rnn2 = nn.seqGRU(lstm.numHidden[1], lstm.numHidden[2])
    rnn3 = nn.seqGRU(lstm.numHidden[2], lstm.numHidden[3])
    rnn4 = nn.seqGRU(lstm.numHidden[3], lstm.numHidden[4])
    rnn5 = nn.seqGRU(lstm.numHidden[4], lstm.numHidden[5])
  end

  self.model:add(rnn1)
  if batchNorm == 1 then
    self.model:add(nn.Sequencer(nn.NormStabilizer()))
  end

  self.model:add(rnn2)
  if batchNorm == 1 then
    self.model:add(nn.Sequencer(nn.NormStabilizer()))
  end

  self.model:add(rnn3)--[[
  if batchNorm == 1 then
    self.model:add(nn.Sequencer(nn.NormStabilizer()))
  end

  self.model:add(rnn4)
  if batchNorm == 1 then
    self.model:add(nn.Sequencer(nn.NormStabilizer()))
  end

  self.model:add(rnn5)
  --]]
  print(self.model)

  return self.model
end

function VideoNet:updateOutput(input)
  return self.model:forward(input)

end

function VideoNet:backward(input, gradOutput)
  return self.model:backward(input, gradOutput)
end


function VideoNet:parameters()
  return self.model:parameters()
end
