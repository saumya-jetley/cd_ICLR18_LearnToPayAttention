require 'nn'

local context_vec_256 = nn.Identity()()

local input = {}
table.insert(input, context_vec_256)

local cv_256_bn = nn.BatchNormalization(256)(context_vec_256)
local cv_256_relu = nn.ReLU(true)(cv_256_bn)
--local cv_256_dp = nn.Dropout(0.3)(cv_256_relu)

local bin_class = nn.Linear(256,10)(cv_256_relu)
--local bin_prob = nn.SoftMax()(bin_class)  -- removing for cross entropy

local output = {}
table.insert(output, bin_class)

local match = nn.gModule(input, output)

return match
