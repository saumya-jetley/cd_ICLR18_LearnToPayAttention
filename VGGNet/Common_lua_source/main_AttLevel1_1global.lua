-- Include the required packages
require 'xlua'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
dofile './provider.lua'
c = require 'trepl.colorize'
model_utils = require 'model_utils' -- to gather params across models
require 'image'

-- default settings
cmd_params = {
	save = 'logs/trials',
	batchSize = 128,
	learningRate = 1,
	learningRateDecay = 1e-7,
	weightDecay = 0.0005,
	momentum = 0.9,
	epoch_step = 25,
	max_epoch = 300,
	model_archi_local = '', --'atten_1_softmax_conv/vgg_conv',
	model_archi_global = '', --'atten_1_softmax_conv/vgg_full',
	model_archi_atten = '', --'atten_1_softmax_conv/atten',
	model_archi_match = '', --'atten_1_softmax_conv/match_singleimagepred',
	model_wts_local = '', --
	model_wts_global = '', -- 
	model_wts_atten = '', --
	model_wts_match '', --
	dataset = '', --'provider.t7',
	num_classes = 0, --10 for cifar-10
	backend = 'nn',
	platformtype = 'cuda',
	gpumode = 1,
	gpu_setDevice = 1,
	mode ='', -- 'train',
}
--[[ If the cmd_prompt has received an updated setting,
update it here, else copy over from default settings --]]
cmd_params = xlua.envparams(cmd_params)

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
function data_aug(input)
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    return input
end

-- Initiation

--1. Data loading
print(c.blue '==>' ..' loading data')
provider = torch.load(cmd_params.dataset)
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()


--2. Model creation
-- At train time
if cmd_params.mode == 'train' then
	model_local = nn.Sequential()
	model_local:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
	model_local:add(cast(dofile(cmd_params.model_archi_local)))
	model_local:get(1).updateGradInput = function(input) return end
	if cmd_params.backend == 'cudnn' then
	   require 'cudnn'
	   cudnn.convert(model_local:get(2), cudnn)
	end

	model_global = nn.Sequential()
	model_global:add(cast(dofile(cmd_params.model_archi_global)))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(model_global:get(1), cudnn)
	end

	model_atten = nn.Sequential()
	model_atten:add(cast(dofile(cmd_params.model_archi_atten)))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(model_atten:get(1),cudnn)
	end

	model_match = nn.Sequential()
	model_match:add(cast(dofile(cmd_params.model_archi_match)))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(model_match:get(1), 'cudnn')
	end
	
	model_all = {}
	table.insert(model_all, model_archi_local)
	table.insert(model_all, model_archi_global)
	table.insert(model_all, model_archi_atten)
	table.insert(model_all, model_archi_match)

	parameters,gradParameters = model_utils.combine_all_parameters(model_all)
	print(parameters:size())


-- At test time
elseif cmd_params.mode == 'test' then
	model_wts_local = torch.load(cmd_params.model_wts_local)
	model_local = nn.Sequential()
	model_local:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
	model_local:add(model_wts_local)

	model_wts_global = torch.load(cmd_params.model_wts_global)
	model_global = nn.Sequential()
	model_global:add(model_wts_global)

	model_wts_atten = torch.load(cmd_params.model_wts_atten)
	model_atten = nn.Sequential()
	model_atten:add(model_wts_atten)

	model_wts_match = torch.load(cmd_params.model_wts_match)
	model_match = nn.Sequential()
	model_match:add(model_wts_match)
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
    model_local:training() 
    model_global:training() 
    model_atten:training() 
    model_match:training()

    epoch = epoch or 1
    
    if epoch % cmd_params.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. cmd_params.batchSize .. ']')
 

    local targets = cast(torch.FloatTensor(cmd_params.batchSize))
    local indices = torch.randperm(provider.trainData.data:size(1)):long():split(cmd_params.batchSize)
    indices[#indices] = nil -- remove last element so that all the batches have equal size

    local tic = torch.tic()
    -- the entire epoch run --
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)
        inputs = provider.trainData.data:index(1,v)
        inputs = data_aug(inputs)
        targets:copy(provider.trainData.labels:index(1,v))      
        ----------------------------------------------------------------------- 
        local feval = function(x)
              if x ~= parameters then parameters:copy(x) end
                  gradParameters:zero()
                  local lfeat = model_local:forward(inputs)
                  local gfeat = model_global:forward(lfeat)
                  local att_con = model_atten:forward({lfeat,gfeat})
                  local prediction = model_match:forward(att_con[2])         
                  
                  local err = criterion:forward(prediction, targets)
            
		  local df_pred = criterion:backward(prediction, targets)
                  local df_context = model_match:backward({att_con[2]}, df_pred)
                  local df_feat = model_atten:backward({lfeat,gfeat}, {torch.rand(att_con[1]:size()):cuda():fill(0), df_context})         
                  local df_lfeat = model_global:backward(lfeat, df_feat[2])                  
                  model_local:backward(inputs,(df_lfeat+df_feat[1])/2)

                  confusion:batchAdd(prediction, targets)
            
                  return f,gradParameters
        end
        optim.sgd(feval, parameters, optimState)
    end
    confusion:updateValids()
    print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(confusion.totalValid * 100, torch.toc(tic)))
    train_acc = confusion.totalValid * 100

    confusion:zero()
    epoch = epoch + 1
end


----- Testing function ----
function test()
   model_local:evaluate()
   model_global:evaluate() 
   model_atten:evaluate() 
	   model_match:evaluate()   -- disable flips, dropouts and batch normalization

  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,provider.testData.data:size(1),bs do
    local lfeat = model_local:forward(provider.testData.data:narrow(1,i,bs))
    local gfeat = model_global:forward(lfeat)
    local att_con = model_atten:forward({lfeat,gfeat})
    local prediction = model_match:forward(att_con[2])                
    confusion:batchAdd(prediction, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(cmd_params.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(cmd_params.save,cmd_params.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(cmd_params.save,cmd_params.save))
      local f = io.open(cmd_params.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end

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
    file:write(tostring(model_archi_local)..'\n')
    file:write(tostring(model_archi_global)..'\n')
    file:write(tostring(model_archi_atten)..'\n')
    file:write(tostring(model_archi_match)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename_loc = paths.concat(cmd_params.save, 'model_local.net')
    print('==> saving model to '..filename_loc)
    torch.save(filename_loc, model_local:get(2):clearState())
        
    local filename_glo = paths.concat(cmd_params.save, 'model_global.net')
    print('==> saving model to '.. filename_glo)
    torch.save(filename_glo, model_global:get(1):clearState())
        
    local filename_att = paths.concat(cmd_params.save, 'model_atten.net')
    print('==> saving model to '.. filename_att)
    torch.save(filename_att, model_atten:get(1):clearState())
        
    local filename_mat = paths.concat(cmd_params.save, 'model_match.net')
    print('==> saving model to '.. filename_mat)
    torch.save(filename_mat, model_match:get(1):clearState())
        
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
