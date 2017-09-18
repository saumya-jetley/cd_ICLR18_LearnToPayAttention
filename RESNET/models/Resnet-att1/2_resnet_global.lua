require 'nn'


local wrn_conv2 = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  wrn_conv2:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  wrn_conv2:add(nn.SpatialBatchNormalization(nOutputPlane))
  wrn_conv2:add(nn.ReLU(true))
  return wrn_conv2
end
-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

---- Network definition
--[[
wrn_conv2:add(nn.SpatialBatchNormalization(256))
wrn_conv2:add(nn.ReLU(true))
wrn_conv2:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(256,256):add(nn.Dropout(0.4))
wrn_conv2:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(256,256):add(nn.Dropout(0.4))
wrn_conv2:add(MaxPooling(2,2,2,2):ceil())
wrn_conv2:add(nn.View(256))

wrn_full3 = nn.Sequential()
wrn_full3:add(nn.Dropout(0.5))
wrn_full3:add(nn.Linear(256,256))
wrn_conv2:add(wrn_full3)
--]]

wrn_conv2:add(nn.SpatialBatchNormalization(256))
wrn_conv2:add(nn.ReLU(true))
wrn_conv2:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(256,256)
wrn_conv2:add(MaxPooling(2,2,2,2):ceil())  --try replacing with AvgPooling
wrn_conv2:add(nn.View(256))


wrn_full3 = nn.Sequential()
wrn_full3:add(nn.Linear(256,256))
wrn_conv2:add(wrn_full3)


--Initialization from MSR
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

MSRinit(wrn_conv2)

return wrn_conv2

