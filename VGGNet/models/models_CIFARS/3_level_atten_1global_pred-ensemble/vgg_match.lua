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
local bin_class1 = nn.Linear(256, cmd_params.num_classes)(context_vec1_do)

local context_vec2_bn = nn.BatchNormalization(512)(context_vec2)
local context_vec2_relu = nn.ReLU(true)(context_vec2_bn)
local context_vec2_do = nn.Dropout(0.5)(context_vec2_relu)
local bin_class2 = nn.Linear(512, cmd_params.num_classes)(context_vec2_do)

local context_vec3_bn = nn.BatchNormalization(512)(context_vec3)
local context_vec3_relu = nn.ReLU(true)(context_vec3_bn)
local context_vec3_do = nn.Dropout(0.5)(context_vec3_relu)
local bin_class3 = nn.Linear(512, cmd_params.num_classes)(context_vec3_do)

local bin_class = nn.CAddTable()({bin_class1, bin_class2, bin_class3})

--local bin_prob = nn.SoftMax()(bin_class)  -- removing for cross entropy criterion
local output = {}
table.insert(output, bin_class)

local match = nn.gModule(input, output)

return match
