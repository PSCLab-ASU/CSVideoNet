require 'torch'
require 'hdf5'
local utils = require 'util.utils'

local DataHandlerVar = torch.class('DataHandlerVar10')

trainDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_conv5_5_25_train.h5'
valDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_conv5_5_25_val.h5'
--testDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/recon/Feature_conv5_5_25_recon.h5'
testDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/Feature_conv5_5_25_recon.h5'

function DataHandlerVar:__init(args)
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

function DataHandlerVar:loadData()
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

function DataHandlerVar:loadTestData()
  local myFile = hdf5.open(testDataFile, 'r')
  self.test.data = myFile:read('data'):all()
  local totalSamples = torch.floor(self.test.data:size(1) / self.imgParam.batchSize)*self.imgParam.batchSize
  self.test.data = self.test.data[{{1,totalSamples},{},{},{},{}}]
  self.test.label = myFile:read('label'):all()
  self.test.label = self.test.label[{{1,totalSamples},{},{},{},{}}]
  self.test.inputNum = totalSamples

  return self
end

function DataHandlerVar:getBatch(dataset, index)
  local i = 1
  local data1 = torch.FloatTensor()
  local data2 = torch.FloatTensor()
  local data3 = torch.FloatTensor()
  local data4 = torch.FloatTensor()
  local data5 = torch.FloatTensor()
  local data6 = torch.FloatTensor()
  local data7 = torch.FloatTensor()
  local data8 = torch.FloatTensor()
  local data9 = torch.FloatTensor()
  local data10 = torch.FloatTensor()

  local label = torch.FloatTensor()
  local rawData = torch.FloatTensor()
  local rawLabel = torch.FloatTensor()

  while i <= self.imgParam.batchSize do
    rawData = self[dataset].data[index]
    rawLabel = self[dataset].label[index]

    if i == 1 then
      data1 = rawData[{{1}, {}, {}, {}}]  -- [1, numChannels, 1, mea]
      --data2 = rawData[{{2,self.imgParam.seqLength}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data2 = rawData[{{2}, {}, {}, {}}]
      data3 = rawData[{{3}, {}, {}, {}}]
      data4 = rawData[{{4}, {}, {}, {}}]
      data5 = rawData[{{5}, {}, {}, {}}]
      data6 = rawData[{{6}, {}, {}, {}}]
      data7 = rawData[{{7}, {}, {}, {}}]
      data8 = rawData[{{8}, {}, {}, {}}]
      data9 = rawData[{{9}, {}, {}, {}}]
      data10= rawData[{{10}, {}, {}, {}}]

      label = rawLabel

    else
      data1 = data1:cat(rawData[{ {1}, {}, {}, {} }], 1)
      -- data2 = data2:cat(rawData[{ {2,self.imgParam.seqLength}, {}, {}, {1,self.imgParam.measurements2} }], 1)
      data2 = data2:cat(rawData[{ {2}, {}, {}, {} }], 1)
      data3 = data1:cat(rawData[{ {3}, {}, {}, {} }], 1)
      data4 = data1:cat(rawData[{ {4}, {}, {}, {} }], 1)
      data5 = data1:cat(rawData[{ {5}, {}, {}, {} }], 1)
      data6 = data1:cat(rawData[{ {6}, {}, {}, {} }], 1)
      data7 = data1:cat(rawData[{ {7}, {}, {}, {} }], 1)
      data8 = data1:cat(rawData[{ {8}, {}, {}, {} }], 1)
      data9 = data1:cat(rawData[{ {9}, {}, {}, {} }], 1)
      data10 = data1:cat(rawData[{ {10}, {}, {}, {} }], 1)

      label = label:cat(rawLabel, 1)
    end

    i = i + 1
  end

    --[[data1 = data1:reshape(-1, 1, self.imgParam.numChannels, self.imgParam.measurements1)
    data2 = data2:reshape(-1, (self.imgParam.seqLength-1), 1, self.imgParam.measurements2)
    label1 = label1:reshape(-1, 1, self.imgParam.numChannels, self.imgParam.Height, self.imgParam.Width)
    label2 = label2:reshape(-1, (self.imgParam.seqLength-1), self.imgParam.numChannels, self.imgParam.Height, self.imgParam.Width)--]]

    -- batch.data = {data1, data2}
    -- batch.label = {label1, label2}

  return data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, label
end

function DataHandlerVar:getValData()
  return self.val
end
