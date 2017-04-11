require 'torch'
require 'hdf5'
local utils = require 'util.utils'

local DataHandler = torch.class('DataHandler')

trainDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/TrainData_25_20736_int.h5'
valDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/ValData_25_20736_int.h5'
testDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/TestData1.h5'

function DataHandler:__init(args)
  self.train = {}
  self.val = {}
  self.test= {}

  self.inputDim = 0

  self.imgParam = {}
  self.imgParam.batchSize = utils.getArgs(args, 'batchSize')
  self.imgParam.seqLength = utils.getArgs(args, 'seqLength')
  self.imgParam.numChannels = utils.getArgs(args, 'numChannels')
  self.imgParam.Height = utils.getArgs(args, 'Height')
  self.imgParam.Width = utils.getArgs(args, 'Width')
  self.imgParam.measurements = utils.getArgs(args, 'measurements')
end

function DataHandler:loadData()
  local myFile = hdf5.open(trainDataFile, 'r')
  self.train.data = myFile:read('data'):all()

  local totalSamples = torch.floor(self.train.data:size(1) / self.imgParam.batchSize)*self.imgParam.batchSize
  self.train.data = self.train.data[{{1,totalSamples},{},{},{},{}}]
  self.train.label = myFile:read('label'):all()
  self.train.label = self.train.label[{{1,totalSamples},{},{},{},{}}]

  self.train.inputNum = totalSamples

  myFile = hdf5.open(valDataFile, 'r')
  self.val.data = myFile:read('data'):all()
  totalSamples = torch.floor(self.val.data:size(1) / self.imgParam.batchSize)*self.imgParam.batchSize
  self.val.data = self.val.data[{{1,totalSamples},{},{},{},{}}]
  self.val.label = myFile:read('label'):all()
  self.val.label = self.val.label[{{1,totalSamples},{},{},{},{}}]

  self.val.inputNum = totalSamples

  --local myFile = hdf5.open(testDataFile, 'r')
  --self.test.data = myFile:read('data'):all()
  --self.test.label = myFile:read('label'):all()
  --self.test.inputNum = self.test.data:size(1)

  return self
end

function DataHandler:getBatch(dataset, index)
  local batch = { data = torch.Tensor(self.imgParam.batchSize * self.imgParam.seqLength,
                                       self.imgParam.numChannels, 1, self.imgParam.measurements):fill(0),
                  label = torch.Tensor(self.imgParam.batchSize * self.imgParam.seqLength,
                                       self.imgParam.numChannels, self.imgParam.Height, self.imgParam.Width):fill(0),

  }
  local i = 1

  while i <= self.imgParam.batchSize do
    batch.data[{ {(i-1) * self.imgParam.seqLength + 1, i * self.imgParam.seqLength}, {}, {}, {} }] = self[dataset].data[index]
    batch.label[{ {(i-1) * self.imgParam.seqLength + 1, i * self.imgParam.seqLength}, {}, {}, {} }] = self[dataset].label[index]
    i = i + 1
  end

  return batch
end

function DataHandler:getValData()
return self.val
end
