require 'cudnn'
local model = {}
-- warning: module 'data [type HDF5Data]' not found
table.insert(model, {'torch_view', nn.View(-1):setNumInputDims(3)})
table.insert(model, {'fc1', nn.Linear(204, 10240)})
-- warning: module 'reshape [type Reshape]' not found
table.insert(model, {'conv1', cudnn.SpatialConvolution(10, 128, 1, 1, 1, 1, 0, 0, 1)})
table.insert(model, {'relu1', cudnn.ReLU(true)})
table.insert(model, {'conv2', cudnn.SpatialConvolution(128, 64, 1, 1, 1, 1, 0, 0, 1)})
table.insert(model, {'relu2', cudnn.ReLU(true)})
table.insert(model, {'conv3', cudnn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1, 1)})
table.insert(model, {'relu3', cudnn.ReLU(true)})
table.insert(model, {'conv4', cudnn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1, 1)})
table.insert(model, {'relu4', cudnn.ReLU(true)})
table.insert(model, {'conv5', cudnn.SpatialConvolution(32, 16, 3, 3, 1, 1, 1, 1, 1)})
table.insert(model, {'relu5', cudnn.ReLU(true)})
table.insert(model, {'conv6', cudnn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1, 1)})
table.insert(model, {'relu6', cudnn.ReLU(true)})
table.insert(model, {'conv7', cudnn.SpatialConvolution(16, 1, 3, 3, 1, 1, 1, 1, 1)})
-- warning: module 'trainLoss [type EuclideanLoss]' not found
return model