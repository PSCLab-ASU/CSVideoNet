require 'image'
utils = require 'util.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(kwargs)
  -- 2 nested tables to store the info of the datasets
  self.splits = {
    train = {},
    test = {},
    val = {}
  }

  self.splits.train.list = utils.getKwarg(kwargs, 'trainList')
  self.splits.test.list = utils.getKwarg(kwargs, 'testList')
  self.splits.val.list = utils.getKwarg(kwargs, 'valList')

  self.opt = {}
  self.opt.seqLength = utils.getKwarg(kwargs, 'seqLength')
  self.opt.imageType = utils.getKwarg(kwargs, 'imageType')
  self.opt.scaledHeight = utils.getKwarg(kwargs, 'scaledHeight')
  self.opt.scaledWeight = utils.getKwarg(kwargs, 'scaledWidth')
  self.opt.batchSize = utils.getKwarg(kwargs, 'batchSize')
  self.opt.cuda = utils.getKwarg(kwargs, 'cuda')

  local loadImageOpt = {
    seqLength = self.opt.seqLength,
    imageTpe = self.opt.imageType,
    scaledHeight = self.opt.scaledHeight,
    scaledWidth = self.opt.scaledWidth
  }

  for split, _ in pairs(self.splits) do
    self.splits[split].index = 1
    self.splits[split].file = paths.basename(self.splits[split].list)
    -- read all the files under the specific directories
    currentPath = self.splits[split].file
    folderNames = {}
    for folderName in paths.files(currentPath) do
      if fileName:find('$') then
          self.splits[split].paths.insert(folderNames, paths.concat(currentPath, folderName))
      end
    end

    self.splits[split].count = #self.splits[split].paths
    self.splits.tradin.shuffle = torch.randperm(self.splits.train.count)

    utils.printTime("Geting train data mean frame")
    self.splits.train.mean = getMeanTrainingImage(self.splits.train.paths,
                                                loadImagesOpt)
  end                                          
end

function DataLoader:nextBatch(split)
  local videoData = {}
  local frameLabel = {}

  while self.splits[split].index <= self.splits[split] and
    #videoData < self.opt.batchSize do

    local index
    if split == 'train' then
      index = self.splits[split].shuffle[self.splits[split].index]
    else
      index = self.splits[split].index
    end
    local videoPath = self.splits[split].paths[index]
    local videoLabel = self.splits[split].paths[index]

    local videoFilename = paths.basename(videoPath)

    for i = 1, self.opt.seqLength do
      local frame = image.load(string.format(videopath .. 'frame%d.%s', i, self.opt.imageType), self.opt.numChannels, 'double')
    end
  end
end

function getMeanTrainingImage(videoPaths, opt)
  local means = {0, 0, 0}
  local numFrames = 0

  for _, videoPath in pairs(videoPaths) do
    -- get video file
    local videoFilename = paths.basename(videoPath)

    -- get path of dumped frames for video
    videoPath = videoFilename .. '_frames'

    -- check if this video qualified to be read (had opt.seqLength or more frames)
    if paths.dirp(videoPath) then
      numFrames = numFrames + opt.seqLength
      local framePath = paths.concat(videoPath, 'frame%d.' .. opt.imageType)
      for i = 1, opt.seqLength do
        -- local frame = image.load(framePath % i, opt.numChannels, 'double')
        local frame = image.load(string.format(videoPath .. '/frame%d.%s', i, opt.imageType), opt.numChannels, 'double')
        frame = image.scale(frame, opt.scaledWidth, opt.scaledHeight) -- scales with string 'WxH', outputs channels x height x width
        for j = 1, opt.numChannels do
          means[j] = means[j] + frame[j]:sum() / (opt.scaledWidth * opt.scaledHeight)
        end
      end
    end
  end

  local meanImage = torch.Tensor(opt.numChannels, opt.scaledHeight, opt.scaledHeight)
  for i = 1, opt.numChannels do
    meanImage[i]:fill(means[i] / numFrames)
  end

  return meanImage
end
