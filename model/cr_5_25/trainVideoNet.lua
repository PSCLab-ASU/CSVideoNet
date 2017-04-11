require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'model.VideoNet'
require 'util.DataHandler'

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

-- Optimization options
cmd:option('-numEpochesTrain', 1000)
cmd:option('-learningRate', 1e-4)
cmd:option('-momentum', 0.9)
cmd:option('-lrDecayFactor', 0.5)
cmd:option('-lrDecayEvery', 3)
cmd:option('-weightDecay', 0);

-- Output options
cmd:option('-printEvery', 10)
cmd:option('-checkpointEvery', 4000)
cmd:option('-checkpointName', 'checkpoints/checkpoint')

cmd:option('-cuda', 1)
cmd:option('-gpu', 0)
cmd:option('-backend', 'nn')
cmd:option('-cr', 5)
cmd:option('-rnnModel', 'LSTM')
cmd:option('-init_from', '')
cmd:option('-epochStart', '')
--checkpoints/checkpointsVideoNet_5_25_2/checkpoint_0.0001_ep4_10000.t7

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
  require 'cunn'
  require 'cudnn'
  opt.dtype = 'torch.CudaTensor'
  cutorch.setDevice(opt.gpu+1)
  opt.dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
else
  print(string.format('Running in CPU mode'))
end

-- load dataset
utils.printTime("Initializing DataLoader")
local loader = DataHandler(opt)
loader = loader:loadData()

-- initialize model
utils.printTime("Initializing videoNet")

if opt.init_from ~= '' then
  print('Initializing from ', opt.init_from)
  model = nn.Sequential()
  local checkpoint = torch.load(opt.init_from)
  model = checkpoint.model
  --model = checkpoint.model:type(dtype)
else
  model = nn.VideoNet(opt):type(opt.dtype)
end

if opt.epochStart == '' then
  opt.epochStart = 1
end

local params, gradParams = model:getParameters()
print(model)
model:training()
model = model:cuda()

local cri = nn.MSECriterion()
cri.sizeAverage = false
local criterion = nn.SequencerCriterion(cri, true):type(opt.dtype)
criterion:cuda()

function train(model)
  utils.printTime(string.format("Starting training for %d epochs", opt.numEpochesTrain))

  local trainLossHistory = {}
  local valLossHistory = {}
  local testLossHistory = {}

  local optimState = {
    learningRate = opt.learningRate,
    --learningRateDecay = opt.lrDecayFactor,
    --momentum = opt.momentum,
   -- dampening = 0.0,
    --weightDecay = opt.weightDecay
  }

  -- For each epoch training
  for iEpochs = opt.epochStart, opt.numEpochesTrain do
    local epochLoss = {}

    if iEpochs % opt.lrDecayEvery == 0 then
      local oldLearningRate = optimState.learningRate
      optimState.learningRate = oldLearningRate * opt.lrDecayFactor
    end

    local loopCount = 1

    dataInitialIndex = torch.range(1, loader.train.inputNum, opt.batchSize)
    numBatches = dataInitialIndex:size(1)
    dataIndex = torch.randperm(numBatches)

    for i = 1, numBatches do
      local data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, label = loader:getBatch('train', dataInitialIndex[dataIndex[i]])

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

      local function feval(x)
        if x ~= params then
          params:copy(x)
        end
        gradParams:zero()

        local modelOut = model:forward({data1, data2, data3, data4, data5, data6, data7, data8, data9, data10})
        --  local label_view = label:view(opt.seqLength, opt.numChannels, opt.Height*opt.Width)
        local frameLoss = criterion:forward(modelOut, label)
        local gradOutput = criterion:backward(modelOut, label)
        local gradModel = model:backward({data1, data2, data3, data4, data5, data6, data7, data8, data9, data10}, gradOutput)

        collectgarbage()
        return frameLoss, gradParams
        -- gradient clamp ???????????????
      end

      local _, loss = optim.adam(feval, params, optimState)
      table.insert(epochLoss, loss[1])

      if((opt.printEvery > 0 and loopCount % opt.printEvery == 0) or loopCount <= 10) then
        local errEveryPixel = loss[1] / (opt.batchSize * opt.numChannels * opt.Width * opt.Height)
        utils.printTime(string.format("Epoch %d, sample %d, training loss: %f, average loss per pixel: %f, current learning rate: %f",
            iEpochs, loopCount, loss[1]/opt.batchSize, errEveryPixel, optimState.learningRate))
      end

      if (opt.checkpointEvery > 0 and loopCount % opt.checkpointEvery == 0) or i == numBatches then
        model:evaluate()
        local valLoss = test(criterion)
        print(optimState)
        model:training()
      end

      if (opt.checkpointEvery > 0 and loopCount % opt.checkpointEvery == 0) or i == numBatches then

        local checkpoint = {
        opt = opt
        --trainLossHistory = trainLossHistory,
        --valLossHistory = valLossHistory
        }

        local filename
        if i == opt.numEpochs then
          filename = string.format('%s_%s_ep%s_%s.t7', opt.checkpointName, opt.learningRate, iEpochs, 'final')
        else
          filename = string.format('%s_%s_ep%s_%d.t7', opt.checkpointName, opt.learningRate, iEpochs, loopCount)
        end

        -- Make sure the output directory exists before we try to write it
        paths.mkdir(paths.dirname(filename))

    --    Cast model to float so it can be used on CPU
    --    model:float()
        checkpoint.model = model
        torch.save(filename, checkpoint)

        -- Cast model back so that it can continue to be used
    --    model:type(opt.dtype)
        utils.printTime(string.format("Saved checkpoint model and opt at %s", filename))
      end

      loopCount = loopCount + 1
      collectgarbage()

    end

    epochLoss =  torch.mean(torch.Tensor(epochLoss))

    table.insert(trainLossHistory, epochLoss)

    utils. printTime(string.format("Epoch %d training loss: %f", iEpochs, epochLoss))

    collectgarbage()
  end
end

function test(criterion)
  utils.printTime(string.format("Starting testing on %s dataset", 'val'))

  local evalData = {}
  evalData.predictedLabels = {}
  evalData.trueLabels = {}

  local totalLoss = 0

  local loopCount = 1

  local dataInitialIndex = torch.range(1, loader.val.inputNum, opt.batchSize)
  local numBatches = dataInitialIndex:size(1)
  local dataIndex = torch.randperm(numBatches)

  for i = 1, numBatches do
    local data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, label= loader:getBatch('val', dataInitialIndex[dataIndex[i]])

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

    local valModelOut = model:forward({data1, data2, data3, data4, data5, data6, data7, data8, data9, data10})
    local loss = criterion:forward(valModelOut, label)
    totalLoss = totalLoss + loss

    if((opt.printEvery > 0 and loopCount % opt.printEvery == 0) or loopCount <= 10) then
        local errEveryPixel = loss / (opt.batchSize * opt.numChannels * opt.Width * opt.Height)
        utils.printTime("----------------------------------------")
        utils.printTime(string.format("Valdation: sample %d, Valdation loss: %f, average loss per pixel: %f, current learning rate: %f", loopCount, loss/opt.batchSize, errEveryPixel, opt.learningRate))
    end

    loopCount = loopCount + 1
  end

  local avgLoss = totalLoss / ( (loopCount-1) * opt.batchSize)
  utils.printTime("+++++++++++++++++++++++++++++++++++++++++++++++")

  local errEveryPixel = avgLoss / (opt.batchSize * opt.numChannels * opt.Width * opt.Height)
  utils.printTime(string.format("Valdation: sample %d, average validation loss: %f, average loss per pixel: %f", loopCount, avgLoss, errEveryPixel))

  collectgarbage()
  return loss
end

train(model)
