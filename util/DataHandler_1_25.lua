require 'torch'
require 'hdf5'
local utils = require 'util.utils'

local DataHandler_1_25 = torch.class('DataHandler_1_25')

trainDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/TrainData_1_25_10172016.h5'
valDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/ValData_1_25_10172016.h5'
--testDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/recon/Feature_conv5_5_25_recon.h5'
testDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/TestData_1_25_10172016.h5'

function DataHandler_1_25:__init(args)
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
  self.imgParam.measurements1 = utils.getArgs(args, 'measurements1')
  self.imgParam.measurements2 = utils.getArgs(args, 'measurements2')
end

function DataHandler_1_25:loadData()
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

  return self
end

function DataHandler_1_25:loadTestData()
  local myFile = hdf5.open(testDataFile, 'r')
  self.test.data = myFile:read('data'):all()
  local totalSamples = torch.floor(self.test.data:size(1) / self.imgParam.batchSize)*self.imgParam.batchSize
  self.test.data = self.test.data[{{1,totalSamples},{},{},{},{}}]
  self.test.label = myFile:read('label'):all()
  self.test.label = self.test.label[{{1,totalSamples},{},{},{},{}}]
  self.test.inputNum = totalSamples

  return self
end

function DataHandler_1_25:getBatch(dataset, index)
  local i = 1
  local data1 = torch.FloatTensor()
  local data2 = torch.FloatTensor()
  local label = torch.FloatTensor()
  local rawData = torch.FloatTensor()
  local rawLabel = torch.FloatTensor()

  for i = 1,self.imgParam.batchSize do
    rawData = self[dataset].data[index+i-1]
    rawLabel = self[dataset].label[index+i-1]

    if i==1 then
      data1 = rawData[{{1}, {}, {}, {}}]  -- [1, numChannels, 1, mea]
      data2 = rawData[{{2}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data3 = rawData[{{3}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data4 = rawData[{{4}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data5 = rawData[{{5}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data6 = rawData[{{6}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data7 = rawData[{{7}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data8 = rawData[{{8}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data9 = rawData[{{9}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data10 = rawData[{{10}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]

      label = rawLabel

    else
      data1 = data1:cat(rawData[{ {1}, {}, {}, {} }], 1)
      data2 = data2:cat(rawData[{ {2,self.imgParam.seqLength}, {}, {}, {1,self.imgParam.measurements2} }], 1)

      label = label:cat(rawLabel, 1)
    end
  end

    --data1 = data1:reshape(-1, 1, self.imgParam.numChannels, self.imgParam.measurements1)
    --data2 = data2:reshape(-1, (self.imgParam.seqLength-1), 1, self.imgParam.measurements2)
    --label1 = label1:reshape(-1, 1, self.imgParam.numChannels, self.imgParam.Height, self.imgParam.Width)
    --label2 = label2:reshape(-1, (self.imgParam.seqLength-1), self.imgParam.numChannels, self.imgParam.Height, self.imgParam.Width)

    -- batch.data = {data1, data2}
    -- batch.label = {label1, label2}

  return data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, label
end

function DataHandler_1_25:getValData()
  return self.val
end
