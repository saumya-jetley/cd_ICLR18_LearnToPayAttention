require 'nn'

local context_vec_128 = nn.Identity()()
local context_vec_256 = nn.Identity()()

local input = {}
table.insert(input, context_vec_128)
table.insert(input, context_vec_256)

local cv_128_bn = nn.BatchNormalization(128)(context_vec_128)
local cv_128_relu = nn.ReLU(true)(cv_128_bn)
local cv_128_dp = nn.Dropout(0.3)(cv_128_relu)

local cv_256_bn = nn.BatchNormalization(256)(context_vec_256)
local cv_256_relu = nn.ReLU(true)(cv_256_bn)
local cv_256_dp = nn.Dropout(0.3)(cv_256_relu)

local cv_all = nn.JoinTable(2)({cv_128_dp, cv_256_dp})
local bin_class = nn.Linear(384,100)(cv_all)


local output = {}
table.insert(output, bin_class)

local match = nn.gModule(input, output)

return match
