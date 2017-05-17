require 'nn'


local vgg_conv_1 = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg_conv_1:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg_conv_1:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg_conv_1:add(nn.ReLU(true))
  return vgg_conv_1
end
-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

---- Network definition

ConvBNReLU(3,64):add(nn.Dropout(0.3))
ConvBNReLU(64,64)

ConvBNReLU(64,128):add(nn.Dropout(0.4))
ConvBNReLU(128,128)

ConvBNReLU(128,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256)
vgg_conv_1:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(256,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)

----- these will be the 4 times downsamples local features -----

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

MSRinit(vgg_conv_1)

return vgg_conv_1
