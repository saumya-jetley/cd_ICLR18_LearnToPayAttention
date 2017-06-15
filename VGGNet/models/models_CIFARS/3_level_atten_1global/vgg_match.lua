require 'nn'

local context_vec1 = nn.Identity()()
local context_vec2 = nn.Identity()()
local context_vec3 = nn.Identity()()

local input = {}
table.insert(input, context_vec1)
table.insert(input, context_vec2)
table.insert(input, context_vec3)

local context_vec1_bn = nn.BatchNormalization(256)(context_vec1)
local context_vec1_relu = nn.ReLU(true)(context_vec1_bn)
local context_vec1_do = nn.Dropout(0.5)(context_vec1_relu)

local context_vec2_bn = nn.BatchNormalization(512)(context_vec2)
local context_vec2_relu = nn.ReLU(true)(context_vec2_bn)
local context_vec2_do = nn.Dropout(0.5)(context_vec2_relu)

local context_vec3_bn = nn.BatchNormalization(512)(context_vec3)
local context_vec3_relu = nn.ReLU(true)(context_vec3_bn)
local context_vec3_do = nn.Dropout(0.5)(context_vec3_relu)

local context_tot = nn.JoinTable(2)({context_vec1_do,context_vec2_do,context_vec3_do})

local bin_class = nn.Linear(1280, cmd_params.num_classes)(context_tot)
--local bin_prob = nn.SoftMax()(bin_class)  -- removing for cross entropy
local output = {}
table.insert(output, bin_class)

local match = nn.gModule(input, output)

return match
