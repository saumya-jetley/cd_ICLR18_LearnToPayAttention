require 'nn'

local context_vec1 = nn.Identity()()

local input = {}
table.insert(input, context_vec1)

local context_vec1_bn = nn.BatchNormalization(512)(context_vec1)
local context_vec1_relu = nn.ReLU(true)(context_vec1_bn)
local context_vec1_do = nn.Dropout(0.5)(context_vec1_relu)

local bin_class = nn.Linear(512, cmd_params.num_classes)(context_vec1_do)

local output = {}
table.insert(output, bin_class)

local match = nn.gModule(input, output)

return match
