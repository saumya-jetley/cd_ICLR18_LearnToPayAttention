require 'nn'

local vgg_full = nn.Sequential()
vgg_full:add(nn.Dropout(0.5))
vgg_full:add(nn.Reshape(32768))   ---- batch_size is automatically taken care of
vgg_full:add(nn.Linear(32768,512))   -- 512*8*8 ASS 
vgg_full:add(nn.BatchNormalization(512))
vgg_full:add(nn.ReLU(true))
vgg_full:add(nn.Linear(512,512))

--vgg_full:add(nn.ReLU(true))  ---  single_im_simple
--vgg_full:add(nn.Linear(512,10)) ---  single_im_simple

return vgg_full
