require 'nngraph'

local lfeat = nn.Identity()()
local gfeat = nn.Identity()()


local atten_input = {}
table.insert(atten_input, lfeat)
table.insert(atten_input, gfeat)

--lfeat = nn.SpatialBatchNormalization(64)(lfeat)
--lfeat = nn.ReLU(true)(lfeat)

local gfeat_bn = nn.BatchNormalization(256)(gfeat)
local gfeat_bn_relu = nn.ReLU(true)(gfeat_bn)
local gfeat_bn_relu_dp = nn.Dropout(0.5)(gfeat_bn_relu)
local gfeat_low = nn.Linear(256,64)(gfeat_bn_relu_dp)

local gfeat_repl1 = nn.Replicate(32,2,1)(gfeat_low) --32 = spatial dimension 1 of lfeat  --ASS
local gfeat_repl2 = nn.Replicate(32,2,1)(gfeat_repl1) --32 = spatial dimension 2 of lfeat  --ASS

local glfeat_add = nn.CAddTable()({gfeat_repl2, lfeat})
local att_mag = nn.SpatialConvolution(64,1,1,1,1,1,0,0)(glfeat_add)  --ASS
local att_mag_reshaped = nn.Reshape(32*32)(att_mag)	--ASS
local att_mag_smax = nn.SoftMax()(att_mag_reshaped)

local att_2d_mask = nn.Reshape(32,32)(att_mag_smax) --ASS

local att_retrans = nn.Replicate(64,2,1)(att_mag_smax)  --ASS

local lfeat_reshaped = nn.Reshape(64, 1024)(lfeat)  --ASS [feature_size x no_of_nodes]
local lfeat_retrans = nn.Transpose({2,3})(lfeat_reshaped)

local convex_comb = nn.CMulTable()({lfeat_retrans, att_retrans})
local context_vec = nn.Sum(1,2)(convex_comb)

local atten_output = {}
table.insert(atten_output, att_2d_mask)
table.insert(atten_output, context_vec)


local atten = nn.gModule(atten_input,atten_output)

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(atten)


return atten
