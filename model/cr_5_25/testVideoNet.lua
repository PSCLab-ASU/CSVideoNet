require 'torch'
require 'nn'
require 'image'
require 'model.VideoNet'
require 'util.DataHandler'
require 'mattorch'

local utils = require 'util.utils'

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-batchNorm', 1)
cmd:option('-batchSize', 80)
cmd:option('-Width', 32)
cmd:option('-Height', 32)
cmd:option('-numChannels', 1)
cmd:option('-measurements1', 204)
cmd:option('-measurements2', 40)

-- Model options
cmd:option('-dropout', 0.5)
cmd:option('-seqLength', 10)

-- Output options
cmd:option('-printEvery', 10)
cmd:option('-checkpointEvery', 5000)
cmd:option('-checkpoint', 'checkpoints/checkpoint_0.0001_ep12_4000.t7')

cmd:option('-cuda', '1')
cmd:option('-gpu', '3')
cmd:option('-backend', 'cudnn')

local opt = cmd:parse(arg)
for k, v in pairs(opt) do
  if tonumber(v) then
    opt[k] = tonumber(v)
  end
end

-- set GPU
opt.dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.cuda == 1 then
  require 'cutorch'
  require 'cudnn'
  require 'cunn'
  opt.dtype = 'torch.CudaTensor'
  cutorch.setDevice(opt.gpu+1)
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
else
  print(string.format('Running in CPU mode'))
end

-- load dataset
utils.printTime("Initializing DataLoader")
local loader = DataHandler(opt)
loader = loader:loadTestData()

-- initialize model
utils.printTime("Initializing VideoNet")
model = nn.Sequential()
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
--model:testing()

local cri = nn.MSECriterion()
cri.sizeAverage = false
local criterion = nn.SequencerCriterion(cri, true):type(opt.dtype)
criterion:cuda()

function test(model)
  collectgarbage()
  utils.printTime(string.format("Starting testing on %s dataset", 'test'))

  local evalData = {}
  evalData.predictedLabels = {}
  evalData.trueLabels = {}

  local totalLoss = 0

  local loopCount = 1

  local dataInitialIndex = torch.range(1, loader.test.inputNum, opt.batchSize)
  local numBatches = dataInitialIndex:size(1)
  local dataIndex = torch.range(1, numBatches)

  local matModelOut = torch.DoubleTensor(numBatches*opt.batchSize, opt.seqLength, opt.numChannels, opt.Height*opt.Width)
  local matLabelOut = torch.DoubleTensor(numBatches*opt.batchSize, opt.seqLength, opt.numChannels, opt.Height*opt.Width)
  for i = 1, numBatches do
    local data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, label = loader:getBatch('test', dataInitialIndex[dataIndex[i]])
    if opt.cuda == 1 then
      data1 = data1:cuda()
      data2 = data2:cuda()
      data3 = data3:cuda()
      data4 = data4:cuda()
      data5 = data5:cuda()
      data6 = data6:cuda()
      data7 = data7:cuda()
      data8 = data8:cuda()
      data9 = data9:cuda()
      data10 = data10:cuda()
      label = label:cuda()
    end

    local testModelOut = model:forward({data1, data2, data3, data4, data5, data6, data7, data8, data9, data10})
    
    local loss = criterion:forward(testModelOut, label)
    totalLoss = totalLoss + loss
    
    testModelOut = testModelOut:transpose(1,2):float()
    label = label:float():transpose(1,2)
    matModelOut[{ {(i-1)*opt.batchSize+1, i*opt.batchSize},{},{} }] = testModelOut
    matLabelOut[{ {(i-1)*opt.batchSize+1, i*opt.batchSize},{},{} }] = label

    if((opt.printEvery > 0 and loopCount % opt.printEvery == 0) or loopCount <= 10) then
      local errEveryPixel = loss / (opt.batchSize * opt.numChannels * opt.Width * opt.Height)
      utils.printTime("----------------------------------------")
      utils.printTime(string.format("Testing: sample %d, Testing loss: %f, average loss per pixel: %f", loopCount, loss/opt.batchSize, errEveryPixel))
    end

    loopCount = loopCount + 1
  end

  local avgLoss = totalLoss / (loopCount-1)
  utils.printTime("+++++++++++++++++++++++++++++++++++++++++++++++")

  local errEveryPixel = avgLoss / (opt.batchSize * opt.numChannels * opt.Width * opt.Height)
  utils.printTime(string.format("Valdation: sample %d, average testing loss: %f, average loss per pixel: %f", loopCount, avgLoss / opt.batchSize, errEveryPixel))

  -- Save model output and label
  list = {modelOut = matModelOut, labelOut = matLabelOut}
  mattorch.save('./recon/reconResult.mat', list)
  --matio.save('testResult.mat', matModelOut)

  collectgarbage()

  return loss
end


test(model)
