require 'nn'
local model = {}
-- warning: module 'data [type HDF5Data]' not found
table.insert(model, {'torch_view', nn.View(-1):setNumInputDims(3)})
table.insert(model, {'fc1', nn.Linear(204, 1024)})
-- warning: module 'reshape [type Reshape]' not found
table.insert(model, {'conv1', nn.SpatialConvolution(1, 128, 1, 1, 1, 1, 0, 0)})
table.insert(model, {'relu1', nn.ReLU(true)})
table.insert(model, {'conv2', nn.SpatialConvolution(128, 64, 1, 1, 1, 1, 0, 0)})
table.insert(model, {'relu2', nn.ReLU(true)})
table.insert(model, {'conv3', nn.SpatialConvolution(64, 32, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu3', nn.ReLU(true)})
table.insert(model, {'conv4', nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu4', nn.ReLU(true)})
table.insert(model, {'conv5', nn.SpatialConvolution(32, 16, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu5', nn.ReLU(true)})
table.insert(model, {'conv6', nn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1, 1)})
table.insert(model, {'relu6', nn.ReLU(true)})
table.insert(model, {'conv7', nn.SpatialConvolution(16, 1, 3, 3, 1, 1, 1, 1)})
-- warning: module 'trainLoss [type EuclideanLoss]' not found
return model