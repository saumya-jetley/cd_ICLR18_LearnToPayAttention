require 'nn'

local context_vec1 = nn.Identity()()
local context_vec2 = nn.Identity()()

local input = {}
table.insert(input, context_vec1)
table.insert(input, context_vec2)

local context_vec1_bn = nn.BatchNormalization(512)(context_vec1)
local context_vec1_relu = nn.ReLU(true)(context_vec1_bn)
local context_vec1_do = nn.Dropout(0.5)(context_vec1_relu)

local context_vec2_bn = nn.BatchNormalization(512)(context_vec2)
local context_vec2_relu = nn.ReLU(true)(context_vec2_bn)
local context_vec2_do = nn.Dropout(0.5)(context_vec2_relu)

local context_tot = nn.JoinTable(2)({context_vec1_do,context_vec2_do})

local bin_class = nn.Linear(1024, cmd_params.num_classes)(context_tot)

local output = {}
table.insert(output, bin_class)

local match = nn.gModule(input, output)

return match
