require 'nn'


local vgg_conv_2 = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg_conv_2:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg_conv_2:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg_conv_2:add(nn.ReLU(true))
  return vgg_conv_2
end
-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

---- Network definition
vgg_conv_2:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(512,512):add(nn.Dropout(0.4))
vgg_conv_2:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(512,512):add(nn.Dropout(0.4))
vgg_conv_2:add(MaxPooling(2,2,2,2):ceil())
vgg_conv_2:add(nn.View(512))

vgg_full_3 = nn.Sequential()
vgg_full_3:add(nn.Dropout(0.5))
vgg_full_3:add(nn.Linear(512,512))
vgg_conv_2:add(vgg_full_3)

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

MSRinit(vgg_conv_2)

return vgg_conv_2

