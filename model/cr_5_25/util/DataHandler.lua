require 'torch'
require 'hdf5'
local utils = require 'util.utils'

local DataHandler = torch.class('DataHandler')

trainDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/TrainData_5_25_mix_feed_10172016.h5'
valDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/data/ValData_5_25_mix_feed_10172016.h5'
--testDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/recon/Feature_conv5_5_25_recon.h5'
testDataFile = '/home/user/kaixu/myGitHub/VideoReconNet/model/cr_5_25_feed/recon/data/TestData_5_25_mix_feed_10172016.h5'

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
  self.imgParam.measurements1 = utils.getArgs(args, 'measurements1')
  self.imgParam.measurements2 = utils.getArgs(args, 'measurements2')
end

function DataHandler:loadData()
  local myFile = hdf5.open(trainDataFile, 'r')
  self.train.data1 = myFile:read('data1'):all()

  local totalSamples = torch.floor(self.train.data1:size(1) / self.imgParam.batchSize)*self.imgParam.batchSize
  self.train.data1 = self.train.data1[{{1,totalSamples},{},{},{},{}}]

  self.train.data2 = myFile:read('data2'):all()
  self.train.data2 = self.train.data2[{{1,totalSamples},{},{},{},{}}]

  self.train.label = myFile:read('label'):all()
  self.train.label = self.train.label[{{1,totalSamples},{},{},{},{}}]

  self.train.inputNum = totalSamples

  myFile = hdf5.open(valDataFile, 'r')
  self.val.data1 = myFile:read('data1'):all()
  totalSamples = torch.floor(self.val.data1:size(1) / self.imgParam.batchSize)*self.imgParam.batchSize
  self.val.data1 = self.val.data1[{{1,totalSamples},{},{},{},{}}]

  self.val.data2 = myFile:read('data2'):all()
  self.val.data2 = self.val.data2[{{1,totalSamples},{},{},{},{}}]

  self.val.label = myFile:read('label'):all()
  self.val.label = self.val.label[{{1,totalSamples},{},{},{},{}}]

  self.val.inputNum = totalSamples

  return self
end

function DataHandler:loadTestData()
  local myFile = hdf5.open(testDataFile, 'r')
  self.test.data1 = myFile:read('data1'):all()
  totalSamples = torch.floor(self.test.data1:size(1) / self.imgParam.batchSize)*self.imgParam.batchSize
  self.test.data1 = self.test.data1[{{1,totalSamples},{},{},{},{}}]
  self.test.data2 = myFile:read('data2'):all()
  self.test.data2 = self.test.data2[{{1,totalSamples},{},{},{},{}}]
  self.test.label = myFile:read('label'):all()
  self.test.label = self.test.label[{{1,totalSamples},{},{},{},{}}]
  self.test.inputNum = totalSamples

  return self
end

function DataHandler:getBatch(dataset, index)
  local i = 1
  local data1_1 = torch.FloatTensor()
  local data2_1 = torch.FloatTensor()
  local data2_2 = torch.FloatTensor()
  local data2_3 = torch.FloatTensor()
  local data2_4 = torch.FloatTensor()
  local data2_5 = torch.FloatTensor()
  local data2_6 = torch.FloatTensor()
  local data2_7 = torch.FloatTensor()
  local data2_8 = torch.FloatTensor()
  local data2_9 = torch.FloatTensor()
  local data2_10 = torch.FloatTensor()

  local label = torch.FloatTensor()
  local rawData1 = torch.FloatTensor()
  local rawData2 = torch.FloatTensor()
  local rawLabel = torch.FloatTensor()

  for i = 1,self.imgParam.batchSize do
    rawData1 = self[dataset].data1[index+i-1]
    rawData2 = self[dataset].data2[index+i-1]
    rawLabel = self[dataset].label[index+i-1]

    if i==1 then
      data1_1 = rawData1[{{1}, {}, {}, {} }]  -- [1, numChannels, 1, mea]
      data2_1 = rawData2[{{1}, {}, {}, {} }] -- [9, numChannels, 1, mea]
      data2_2 = rawData2[{{2}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data2_3 = rawData2[{{3}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data2_4 = rawData2[{{4}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data2_5 = rawData2[{{5}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data2_6 = rawData2[{{6}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data2_7 = rawData2[{{7}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data2_8 = rawData2[{{8}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data2_9 = rawData2[{{9}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]
      data2_10 = rawData2[{{10}, {}, {}, {1,self.imgParam.measurements2} }] -- [9, numChannels, 1, mea]

      label = rawLabel:view(self.imgParam.seqLength, -1)
      label = label:view(-1, self.imgParam.seqLength, self.imgParam.Height*self.imgParam.Width)
    else
      data1_1 = data1_1:cat(rawData1[{{1}, {}, {}, {} }], 1)
      data2_1 = data2_1:cat(rawData2[{{1}, {}, {}, {} }], 1)
      data2_2 = data2_2:cat(rawData2[{{2}, {}, {}, {1,self.imgParam.measurements2} }], 1)
      data2_3 = data2_3:cat(rawData2[{{3}, {}, {}, {1,self.imgParam.measurements2} }], 1)
      data2_4 = data2_4:cat(rawData2[{{4}, {}, {}, {1,self.imgParam.measurements2} }], 1)
      data2_5 = data2_5:cat(rawData2[{{5}, {}, {}, {1,self.imgParam.measurements2} }], 1)
      data2_6 = data2_6:cat(rawData2[{{6}, {}, {}, {1,self.imgParam.measurements2} }], 1)
      data2_7 = data2_7:cat(rawData2[{{7}, {}, {}, {1,self.imgParam.measurements2} }], 1)
      data2_8 = data2_8:cat(rawData2[{{8}, {}, {}, {1,self.imgParam.measurements2} }], 1)
      data2_9 = data2_9:cat(rawData2[{{9}, {}, {}, {1,self.imgParam.measurements2} }], 1)
      data2_10 = data2_10:cat(rawData2[{{10}, {}, {}, {1,self.imgParam.measurements2} }], 1)

      labelTmp = rawLabel:view(self.imgParam.seqLength, -1)
      labelTmp = rawLabel:view(-1, self.imgParam.seqLength, self.imgParam.Height*self.imgParam.Width)
      label = label:cat(labelTmp, 1)
    end
  end
  label = label:transpose(1,2)

    --data1 = data1:reshape(-1, 1, self.imgParam.numChannels, self.imgParam.measurements)
    --data2 = data2:reshape(-1, (self.imgParam.seqLength-1), 1, self.imgParam.measurements2)
    --label1 = label1:reshape(-1, 1, self.imgParam.numChannels, self.imgParam.Height, self.imgParam.Width)
    --label2 = label2:reshape(-1, (self.imgParam.seqLength-1), self.imgParam.numChannels, self.imgParam.Height, self.imgParam.Width)

    -- batch.data = {data1, data2}
    -- batch.label = {label1, label2}

  local data1 = data1_1
  local data2 = data2_1:cat(data2_2, 4)
  local data3 = data2_1:cat(data2_3, 4)
  local data4 = data2_1:cat(data2_4, 4)
  local data5 = data2_1:cat(data2_5, 4)
  local data6 = data2_1:cat(data2_6, 4)
  local data7 = data2_1:cat(data2_7, 4)
  local data8 = data2_1:cat(data2_8, 4)
  local data9 = data2_1:cat(data2_9, 4)
  local data10 = data2_1:cat(data2_10, 4)

  return data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, label
end

function DataHandler:getValData()
  return self.val
end
