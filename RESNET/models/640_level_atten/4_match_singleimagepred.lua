require 'nn'

local context_vec1 = nn.Identity()()

local input = {}
table.insert(input, context_vec1)

--local cv1_dp = nn.Dropout(0.5)(context_vec1)
--local context_vec2 = nn.Linear(640,640)(cv1_dp)
local cv2_bn = nn.BatchNormalization(640)(context_vec1)
local cv2_bn_relu = nn.ReLU(true)(cv2_bn)
local cv2_bn_relu_dp = nn.Dropout(0.3)(cv2_bn_relu)
local bin_class = nn.Linear(640,100)(cv2_bn_relu_dp)
--local bin_prob = nn.SoftMax()(bin_class)  -- removing for cross entropy

local output = {}
table.insert(output, bin_class)

local match = nn.gModule(input, output)

return match
