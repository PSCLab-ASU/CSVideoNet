require 'torch'
require 'hdf5'
local utils = require 'util.utils'

local DataHandler10 = torch.class('DataHandler10')

trainDataFile1 = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_5_train_10172016.h5'
valDataFile1 = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_5_val_10172016.h5'

trainDataFile2 = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_25_train_10172016.h5'
valDataFile2 = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_25_val_10172016.h5'

trainDataFile3 = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_50_train_10172016.h5'
valDataFile3 = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_50_val_10172016.h5'

trainDataFile4 = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_100_train_10172016.h5'
valDataFile4 = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_100_val_10172016.h5'

testDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_conv5_5_recon.h5'

function DataHandler10:__init(args)
  self.train = {}
  self.val = {}
  self.test= {}

  self.inputDim = 0

  self.cr = utils.getArgs(args, 'cr')
  self.imgParam = {}
  self.imgParam.batchSize = utils.getArgs(args, 'batchSize')
  self.imgParam.seqLength = utils.getArgs(args, 'seqLength')
  self.imgParam.numChannels = utils.getArgs(args, 'numChannels')
  self.imgParam.Height = utils.getArgs(args, 'Height')
  self.imgParam.Width = utils.getArgs(args, 'Width')
  self.imgParam.measurements = utils.getArgs(args, 'measurements')
end

function DataHandler10:loadData()
  if self.cr == 5 then
    myFile = hdf5.open(trainDataFile1, 'r')
  elseif self.cr == 25 then
    myFile = hdf5.open(trainDataFile2, 'r')
  elseif self.cr == 50 then
    myFile = hdf5.open(trainDataFile3, 'r')
  elseif self.cr == 100 then
    myFile = hdf5.open(trainDataFile4, 'r')
  else
    error("invalid cr")
  end
  
  self.train.data = myFile:read('data'):all()

  local totalSamples = torch.floor(self.train.data:size(1) / self.imgParam.batchSize)*self.imgParam.batchSize
  self.train.data = self.train.data[{{1,totalSamples},{},{},{},{}}]
  self.train.label = myFile:read('label'):all()
  self.train.label = self.train.label[{{1,totalSamples},{},{},{},{}}]

  self.train.inputNum = totalSamples

  if self.cr == 5 then
    myFile = hdf5.open(valDataFile1, 'r')
  elseif self.cr == 25 then
    myFile = hdf5.open(valDataFile2, 'r')
  elseif self.cr == 50 then
    myFile = hdf5.open(valDataFile3, 'r')
  elseif self.cr == 100 then
    myFile = hdf5.open(valDataFile4, 'r')
  end
  
  self.val.data = myFile:read('data'):all()
  totalSamples = torch.floor(self.val.data:size(1) / self.imgParam.batchSize)*self.imgParam.batchSize
  self.val.data = self.val.data[{{1,totalSamples},{},{},{},{}}]
  self.val.label = myFile:read('label'):all()
  self.val.label = self.val.label[{{1,totalSamples},{},{},{},{}}]

  self.val.inputNum = totalSamples

  return self
end

function DataHandler10:loadTestData()
  local myFile = hdf5.open(testDataFile, 'r')
  self.test.data = myFile:read('data'):all()
  local totalSamples = torch.floor(self.test.data:size(1) / self.imgParam.batchSize)*self.imgParam.batchSize
  self.test.data = self.test.data[{{1,totalSamples},{},{},{},{}}]
  self.test.label = myFile:read('label'):all()
  self.test.label = self.test.label[{{1,totalSamples},{},{},{},{}}]
  self.test.inputNum = totalSamples

  return self
end

function DataHandler10:getBatch(dataset, index)
  local data = torch.FloatTensor()
  local label = torch.FloatTensor()
  local i = 1

  data = self[dataset].data[index]
  label = self[dataset].label[index]

  while i <= self.imgParam.batchSize-1 do
    data = data:cat(self[dataset].data[index+i],1)
    label = label:cat(self[dataset].label[index+i-1],1)
    i = i + 1
  end

  return data, label
end

function DataHandler10:getValData()
  return self.val
end
