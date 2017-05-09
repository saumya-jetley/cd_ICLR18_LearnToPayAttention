require 'nngraph'

local lfeat = nn.Identity()()
local gfeat = nn.Identity()()


local atten_input = {}
table.insert(atten_input, lfeat)
table.insert(atten_input, gfeat)

local gfeat_repl1 = nn.Replicate(32,2,1)(gfeat) -- spatial dimension 1 of lfeat
local gfeat_repl2 = nn.Replicate(32,2,1)(gfeat_repl1) -- spatial dimension 2 of lfeat

local glfeat_add = nn.CAddTable()({gfeat_repl2, lfeat})
local att_mag = nn.SpatialConvolution(256,1,1,1,1,1,0,0)(glfeat_add)
local att_mag_reshaped = nn.Reshape(32*32)(att_mag)
local att_mag_smax = nn.SoftMax()(att_mag_reshaped)

local att_2d_mask = nn.Reshape(32,32)(att_mag_smax)

local att_retrans = nn.Replicate(256,2,1)(att_mag_smax)

local lfeat_reshaped = nn.Reshape(256, 1024)(lfeat)
local lfeat_retrans = nn.Transpose({2,3})(lfeat_reshaped)

local convex_comb = nn.CMulTable()({lfeat_retrans, att_retrans})
local convex_comb_tr = nn.Transpose({2,3})(convex_comb)
local context_map = nn.Reshape(256,32,32)(convex_comb_tr)

local atten_output = {}
table.insert(atten_output, att_2d_mask)
table.insert(atten_output, context_map)



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
