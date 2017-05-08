require 'nn'

local context_vec1 = nn.Identity()()

local input = {}
table.insert(input, context_vec1)

local context_vec1_relu = nn.ReLU(true)(context_vec1)
local bin_class = nn.Linear(512, opt.num_classes)(context_vec1_relu)   --- expecting 2 vectors of 512 dims

local output = {}
table.insert(output, bin_class)

local match = nn.gModule(input, output)

return match
