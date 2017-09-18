require 'nngraph'

local lfeat = nn.Identity()()

local atten_input = {}
table.insert(atten_input, lfeat)

local gfeat = nn.SpatialAveragePooling(8, 8, 1, 1)(lfeat)  ---ASS   
local gfeat_repl1 = nn.Replicate(8,2,1)(gfeat) --8 = spatial dimension 1 of lfeat  --ASS
local gfeat_repl2 = nn.Replicate(8,2,1)(gfeat_repl1) --8 = spatial dimension 2 of lfeat  --ASS
local gfeat_repl2 = nn.Squeeze()(gfeat_repl2)

local glfeat_add = nn.CAddTable()({gfeat_repl2, lfeat})
local att_mag = nn.SpatialConvolution(640,1,1,1,1,1,0,0)(glfeat_add) ---ASS
local att_mag_reshaped = nn.Reshape(8*8)(att_mag)   ---ASS
local att_mag_smax = nn.SoftMax()(att_mag_reshaped)

local att_2d_mask = nn.Reshape(8,8)(att_mag_smax) ---ASS

local att_retrans = nn.Replicate(640,2,1)(att_mag_smax)  ---ASS
local lfeat_reshaped = nn.Reshape(640, 64)(lfeat)  ---ASS
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