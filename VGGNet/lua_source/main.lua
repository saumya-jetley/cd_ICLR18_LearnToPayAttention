-- Include the required packages
require 'xlua'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
dofile './provider.lua'
c = require 'trepl.colorize'

-- default settings
cmd_params = {
	save = 'logs/trials',
	batchSize = 128,
	learningRate = 1,
	learningRateDecay = 1e-7,
	weightDecay = 0.0005,
	momentum = 0.9,
	epoch_step = '25',
	lr_step = '0.5',
	max_epoch = 300,
	model_archi = '', --'vgg_bn_drop',
	model_wts = '', --'./logs/vgg_withHFLIP/model.net',
	dataset = '', --'provider.t7',
	num_classes = 0, --10 for cifar-10
	backend = 'nn',
	platformtype = 'cuda',
	gpumode = 1,
	gpu_setDevice = 1,
	mode = '', --'train',
}

--[[ If the cmd_prompt has received an updated setting,
update it here, else copy over from default settings --]]
cmd_params = xlua.envparams(cmd_params)

cmd_params.epoch_step = tonumber(cmd_params.epoch_step) or loadstring('return '..cmd_params.epoch_step)()
cmd_params.lr_step = tonumber(cmd_params.lr_step) or loadstring('return '..cmd_params.lr_step)()

-- Setting for the random number generator
local seed = 1234567890
torch.manualSeed(seed)

-- support function - Data casting
function cast(t)
   if cmd_params.platformtype == 'cuda' then
      require 'cunn'
        gpumode = cmd_params.gpumode
        if gpumode==1 then
            cutorch.setDevice(cmd_params.gpu_setDevice)
        end
      return t:cuda()
   elseif cmd_params.platformtype == 'float' then
      return t:float()
   elseif cmd_params.platformtype == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..cmd_params.platformtype)
   end
end

-- support function - Data augmentation
function hflip_aug(input)
      --hflip
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    return input
end

function rot_aug(input, pad)
    -- Rotation
    assert(input:dim() == 4)
    local imsize = input:size(4)
    local padded = nn.SpatialZeroPadding(pad,pad,pad,pad):forward(input)
    local x = torch.random(1,pad*2 + 1)
    local y = torch.random(1,pad*2 + 1)
    local input_rot = padded:narrow(4,x,imsize):narrow(3,y,imsize)
    return input_rot:contiguous()
end


-- Initiation

--1. Data loading
print(c.blue '==>' ..' loading data')
print(cmd_params.dataset)
provider = torch.load(cmd_params.dataset)
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()


--2. Model creation
-- At train time
if cmd_params.mode == 'train' then
	model = nn.Sequential()
	model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
	model:add(cast(dofile(cmd_params.model_archi)))
	model:get(1).updateGradInput = function(input) return end
	if cmd_params.backend == 'cudnn' then
	   require 'cudnn'
	   cudnn.convert(model:get(2), cudnn)
	end
	parameters,gradParameters = model:getParameters()
-- At test time
elseif cmd_params.mode == 'test' then
	model_wts = torch.load(cmd_params.model_wts)
	model = nn.Sequential()
	model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
	model:add(model_wts)
end

--3. Criterion
print(c.blue'==>' ..' setting criterion')
criterion = cast(nn.CrossEntropyCriterion())

--4. Testing and saving
confusion = optim.ConfusionMatrix(cmd_params.num_classes)
print('Will save at '..cmd_params.save)
paths.mkdir(cmd_params.save)
testLogger = optim.Logger(paths.concat(cmd_params.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

--5. Learning settings
print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = cmd_params.learningRate,
  weightDecay = cmd_params.weightDecay,
  momentum = cmd_params.momentum,
  learningRateDecay = cmd_params.learningRateDecay,
}


----- Training function -----
function train()
 
    model:training()
    epoch = epoch or 1
    
    if torch.type(cmd_params.epoch_step) == 'number' and epoch % cmd_params.epoch_step == 0 then 
	optimState.learningRate = optimState.learningRate*cmd_params.lr_step
    elseif torch.type(cmd_params.epoch_step) == 'table' and tablex.find(cmd_params.epoch_step, epoch) then
	optimState.learningRate = optimState.learningRate*cmd_params.lr_step[tablex.find(cmd_params.epoch_step, epoch)]
    end

    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. cmd_params.batchSize .. ']')  

    local targets = cast(torch.FloatTensor(cmd_params.batchSize))
    local indices = torch.randperm(provider.trainData.data:size(1)):long():split(cmd_params.batchSize)
    indices[#indices] = nil -- remove last element so that all the batches have equal size

    local tic = torch.tic()
    -- the entire epoch run --
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)
        inputs = provider.trainData.data:index(1,v)
        inputs = hflip_aug(inputs)
        targets:copy(provider.trainData.labels:index(1,v))
        
        local feval = function(x)
              if x ~= parameters then parameters:copy(x) end
              gradParameters:zero()

              local outputs = model:forward(inputs)
              local f = criterion:forward(outputs, targets)
              local df_do = criterion:backward(outputs, targets)
              model:backward(inputs, df_do)

              confusion:batchAdd(outputs, targets)

              return f,gradParameters
        end
        optim.sgd(feval, parameters, optimState)
    end
    -------------
  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1

end


----- Testing function ----
function test()
  epoch = epoch or 1
  model:evaluate()   -- disable flips, dropouts and batch normalization
  print(c.blue '==>'.." testing")
  local bs = 16
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(cmd_params.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

--[[
    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(cmd_params.save,cmd_params.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(cmd_params.save,cmd_params.save))
      local f = io.open(cmd_params.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end
--]]

    local file = io.open(cmd_params.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(cmd_params.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename = paths.concat(cmd_params.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(2):clearState())
  end

  confusion:zero()
end


---------------------------
------ Main Run ------------
---------------------------
if cmd_params.mode == 'train' then
	for i=1,cmd_params.max_epoch do
	  train()
	  test()
	end
elseif cmd_params.mode == 'test' then
	  test()
end
