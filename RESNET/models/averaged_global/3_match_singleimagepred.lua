require 'nn'

local context_vec1 = nn.Identity()()

local input = {}
table.insert(input, context_vec1)

local bin_class = nn.Linear(640,100)(context_vec1)  ---ASS
--local bin_prob = nn.SoftMax()(bin_class)  -- removing for cross entropy
local output = {}
table.insert(output, bin_class)

local match = nn.gModule(input, output)

return match