require 'nn'

local context_vec1 = nn.Identity()()
local context_vec2 = nn.Identity()()

local input = {}
table.insert(input, context_vec1)
table.insert(input, context_vec2)

local joint_vector = nn.JoinTable(1,1)({context_vec1, context_vec2})
local enc1 = nn.Linear(1024, 1024)(joint_vector)   --- expecting 2 vectors of 512 dims  --ASS
local enc1_relu = nn.ReLU(true)(enc1)
local enc2 = nn.Linear(1024, 32)(enc1_relu)
local enc2_relu = nn.ReLU(true)(enc2)
local bin_class = nn.Linear(32,2)(enc2_relu)
----local bin_prob = nn.SoftMax()(bin_class)  -- removing for cross entropy

local output = {}
table.insert(output, bin_class)

local match = nn.gModule(input, output)

return match