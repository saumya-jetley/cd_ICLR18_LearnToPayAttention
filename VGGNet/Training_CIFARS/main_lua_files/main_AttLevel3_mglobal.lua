-- Include the required packages
require 'xlua'
require 'optim'
require 'nn'
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
	model_archi_local_1 = '1.1_vgg_local',
	model_archi_local_2 = '1.2_vgg_local',
	model_archi_local_3 = '1.3_vgg_local',
	model_archi_global_1 = '2.1_vgg_global'
	model_archi_global_2 = '2.2_vgg_global',
	model_archi_global_3 = '2.3_vgg_global',
	model_archi_atten_1 = '3.1_vgg_atten',
	model_archi_atten_2 = '3.2_vgg_atten',
	model_archi_atten_3 = '3.3_vgg_atten',
	model_archi_match = '4_vgg_match',
	model_wts_local_1 = '1.1_vgg_local',
	model_wts_local_2 = '1.2_vgg_local',
	model_wts_local_3 = '1.3_vgg_local',
	model_wts_global_1 = '2.1_vgg_global'
	model_wts_global_2 = '2.2_vgg_global',
	model_wts_global_3 = '2.3_vgg_global',
	model_wts_atten_1 = '3.1_vgg_atten',
	model_wts_atten_2 = '3.2_vgg_atten',
	model_wts_atten_3 = '3.3_vgg_atten',
	model_wts_match = '4_vgg_match',
	dataset = 'provider.t7',
	num_classes = 10, -- by default set to cifar-10
	backend = 'nn',
	platformtype = 'cuda',
	gpumode = 1,
	gpu_setDevice = 1,
	mode = 'train',
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
	mlocal_1 = nn.Sequential()
	mlocal_1:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
	mlocal_1:add(cast(dofile('models/'..cmd_params.model_archi_local_1..'.lua')))
	mlocal_1:get(1).updateGradInput = function(input) return end
	if cmd_params.backend == 'cudnn' then
	   require 'cudnn'
	   cudnn.convert(mlocal_1:get(2), cudnn)
	end

	mlocal_2 = nn.Sequential()
	mlocal_2:add(cast(dofile('models/'..cmd_params.model_archi_local_2..'.lua')))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(mlocal_2:get(1), cudnn)
	end

	mlocal_3 = nn.Sequential()
	mlocal_3:add(cast(dofile('models/'..cmd_params.model_archi_local_3..'.lua')))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(mlocal_3:get(1), cudnn)
	end
	-------------------------------------------------------------------------------------------------

	mglobal_1 = nn.Sequential()
	mglobal_1:add(cast(dofile('models/'..cmd_params.model_archi_global_1..'.lua')))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(mglobal_1:get(1), cudnn)
	end

	mglobal_2 = nn.Sequential()
	mglobal_2:add(cast(dofile('models/'..cmd_params.model_archi_global_2..'.lua')))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(mglobal_2:get(1), cudnn)
	end

	mglobal_3 = nn.Sequential()
	mglobal_3:add(cast(dofile('models/'..cmd_params.model_archi_global_3..'.lua')))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(mglobal_3:get(1), cudnn)
	end
	-------------------------------------------------------------------------------------------------

	matten_1 = nn.Sequential()
	matten_1:add(cast(dofile('models/'..cmd_params.moel_archi_atten_1..'.lua')))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(matten_1:get(1),cudnn)
	end

	matten_2 = nn.Sequential()
	matten_2:add(cast(dofile('models/'..cmd_params.model_archi_atten_2..'.lua')))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(matten_2:get(1),cudnn)
	end

	matten_3 = nn.Sequential()
	matten_3:add(cast(dofile('models/'..cmd_params.model_archi_atten_3..'.lua')))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(matten_3:get(1),cudnn)
	end
	-------------------------------------------------------------------------------------------------

	mmatch = nn.Sequential()
	mmatch:add(cast(dofile('models/' ..cmd_params.model_archi_match..'.lua')))
	if cmd_params.backend == 'cudnn' then
	    cudnn.convert(mmatch:get(1), 'cudnn')
	end

	-------------------------------------------------------------------------------------------------
	model_all = {}
	table.insert(model_all, mlocal_1)
	table.insert(model_all, mlocal_2)
	table.insert(model_all, mlocal_3)

	table.insert(model_all, mglobal_1)
	table.insert(model_all, mglobal_2)
	table.insert(model_all, mglobal_3)

	table.insert(model_all, matten_1)
	table.insert(model_all, matten_2)
	table.insert(model_all, matten_3)

	table.insert(model_all, mmatch)

	parameters,gradParameters = model_utils.combine_all_parameters(model_all)
	print(parameters:size())

-- At test time
elseif cmd_params.mode == 'test' then

	model_wts_local_1 = torch.load(cmd_params.model_wts_local_1)
	mlocal_1 = nn.Sequential()
	mlocal_1:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
	mlocal_1:add(model_wts_local_1)

	model_wts_local_2 = torch.load(cmd_params.model_wts_local_2)
	mlocal_2 = nn.Sequential()
	mlocal_2:add(model_wts_local_2)

	model_wts_local_3 = torch.load(cmd_params.model_wts_local_3)
	mlocal_3 = nn.Sequential()
	mlocal_3:add(model_wts_local_3)
	-------------------------------------------------------------------------------------------------
	model_wts_global_1 = torch.load(cmd_params.model_wts_global_1)	
	mglobal_1 = nn.Sequential()
	mglobal_1:add(model_wts_global_1)

	model_wts_global_2 = torch.load(cmd_params.model_wts_global_2)
	mglobal_2 = nn.Sequential()
	mglobal_2:add(model_wts_global_2)

	model_wts_global_3 = torch.load(cmd_params.model_wts_global_3)
	mglobal_3 = nn.Sequential()
	mglobal_3:add(model_wts_global_3)
	-------------------------------------------------------------------------------------------------
	model_wts_atten_1 = torch.load(cmd_params.model_wts_atten_1)
	matten_1 = nn.Sequential()
	matten_1:add(model_wts_atten_1)

	model_wts_atten_2 = torch.load(cmd_params.model_wts_atten_2)
	matten_2 = nn.Sequential()
	matten_2:add(model_wts_atten_2)

	model_wts_atten_3 = torch.load(cmd_params.model_wts_atten_3)
	matten_3 = nn.Sequential()
	matten_3:add(model_wts_atten_3)
	-------------------------------------------------------------------------------------------------
	model_wts_match = torch.load(cmd_params.model_wts_match)
	mmatch = nn.Sequential()
	mmatch:add(model_wts_match)
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
    mlocal_1:training() 
    mlocal_2:training() 
    mlocal_3:training() 
    
    mglobal_1:training() 
    mglobal_2:training()
    mglobal_3:training()
    
    matten_1:training() 
    matten_2:training() 
    matten_3:training() 
    
    mmatch:training()

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
                  ---------forward
                  local lfeat_1 = mlocal_1:forward(inputs)           
                  local gfeat_1 = mglobal_1:forward(lfeat_1)                          
                  local att_con_1 = matten_1:forward({lfeat_1,gfeat_1})
        
                  local lfeat_2 = mlocal_2:forward(lfeat_1)           
                  local gfeat_2 = mglobal_2:forward(lfeat_2)                          
                  local att_con_2 = matten_2:forward({lfeat_2,gfeat_2})
        
                  local lfeat_3 = mlocal_3:forward(lfeat_2)           
                  local gfeat_3 = mglobal_3:forward(lfeat_3)                          
                  local att_con_3 = matten_3:forward({lfeat_3,gfeat_3})
        
                  local prediction = mmatch:forward({att_con_1[2], att_con_2[2], att_con_3[2]})         
                  
                  local err = criterion:forward(prediction, targets)            
            
                  ---------backward            
                  local df_pred = criterion:backward(prediction, targets)
                  local df_context = mmatch:backward({att_con_1[2], att_con_2[2], att_con_3[2]}, df_pred)
                  
                  local df_feat_3 = matten_3:backward({lfeat_3,gfeat_3}, {torch.rand(att_con_3[1]:size()):cuda():fill(0), df_context[3]})                               
                  local df_lfeat_3 = mglobal_3:backward(lfeat_3, df_feat_3[2])                  
                  local df_lfeat_chain_2 = mlocal_3:backward(lfeat_2,(df_lfeat_3+df_feat_3[1])/2)
            
                  local df_feat_2 = matten_2:backward({lfeat_2,gfeat_2}, {torch.rand(att_con_2[1]:size()):cuda():fill(0), df_context[2]})                               
                  local df_lfeat_2 = mglobal_2:backward(lfeat_2, df_feat_2[2])                  
                  local df_lfeat_chain_1 = mlocal_2:backward(lfeat_1,(df_lfeat_chain_2 + df_lfeat_2 + df_feat_2[1])/2)
                
                  local df_feat_1 = matten_1:backward({lfeat_1,gfeat_1}, {torch.rand(att_con_1[1]:size()):cuda():fill(0), df_context[1]})                               
                  local df_lfeat_1 = mglobal_1:backward(lfeat_1, df_feat_1[2])                  
                  mlocal_1:backward(inputs,(df_lfeat_1+df_feat_1[1]+df_lfeat_chain_1)/3)

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
  -- disable flips, dropouts and batch normalization
    mlocal_1:evaluate()
    mlocal_2:evaluate()
    mlocal_3:evaluate()
    
    mglobal_1:evaluate() 
    mglobal_2:evaluate() 
    mglobal_3:evaluate() 
    
    matten_1:evaluate() 
    matten_2:evaluate() 
    matten_3:evaluate() 
    
    mmatch:evaluate()

  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,provider.testData.data:size(1),bs do
    local lfeat_1 = mlocal_1:forward(provider.testData.data:narrow(1,i,bs))           
    local gfeat_1 = mglobal_1:forward(lfeat_1)                          
    local att_con_1 = matten_1:forward({lfeat_1,gfeat_1})
        
    local lfeat_2 = mlocal_2:forward(lfeat_1)           
    local gfeat_2 = mglobal_2:forward(lfeat_2)                          
    local att_con_2 = matten_2:forward({lfeat_2,gfeat_2})
        
    local lfeat_3 = mlocal_3:forward(lfeat_2)           
    local gfeat_3 = mglobal_3:forward(lfeat_3)                          
    local att_con_3 = matten_3:forward({lfeat_3,gfeat_3})
        
    local prediction = mmatch:forward({att_con_1[2], att_con_2[2], att_con_3[2]})         
                        
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
    file:write(tostring(mlocal_1)..'\n')
    file:write(tostring(mlocal_2)..'\n')
    file:write(tostring(mlocal_3)..'\n')
        
    file:write(tostring(mglobal_1)..'\n')
    file:write(tostring(mglobal_2)..'\n')
    file:write(tostring(mglobal_3)..'\n')
        
    file:write(tostring(matten_1)..'\n')
    file:write(tostring(matten_2)..'\n')
    file:write(tostring(matten_3)..'\n')
        
    file:write(tostring(mmatch)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename_loc1 = paths.concat(cmd_params.save, 'mlocal_1.net')
    print('==> saving model to '..filename_loc1)
    torch.save(filename_loc1, mlocal_1:get(2):clearState())
    
    local filename_glo1 = paths.concat(cmd_params.save, 'mglobal_1.net')
    print('==> saving model to '.. filename_glo1)
    torch.save(filename_glo1, mglobal_1:get(1):clearState())
    
    local filename_att1 = paths.concat(cmd_params.save, 'matten_1.net')
    print('==> saving model to '.. filename_att1)
    torch.save(filename_att1, matten_1:get(1):clearState())
    
        
    local filename_loc2 = paths.concat(cmd_params.save, 'mlocal_2.net')
    print('==> saving model to '.. filename_loc2)
    torch.save(filename_loc2, mlocal_2:get(1):clearState())
    
    local filename_glo2 = paths.concat(cmd_params.save, 'mglobal_2.net')
    print('==> saving model to '.. filename_glo2)
    torch.save(filename_glo2, mglobal_2:get(1):clearState())    
        
    local filename_att2 = paths.concat(cmd_params.save, 'matten_2.net')
    print('==> saving model to '.. filename_att2)
    torch.save(filename_att2, matten_2:get(1):clearState())
    
    
    local filename_loc3 = paths.concat(cmd_params.save, 'mlocal_3.net')
    print('==> saving model to '.. filename_loc3)
    torch.save(filename_loc3, mlocal_3:get(1):clearState())
    
    local filename_glo3 = paths.concat(cmd_params.save, 'mglobal_3.net')
    print('==> saving model to '.. filename_glo3)
    torch.save(filename_glo3, mglobal_3:get(1):clearState())    
        
    local filename_att3 = paths.concat(cmd_params.save, 'matten_3.net')
    print('==> saving model to '.. filename_att3)
    torch.save(filename_att3, matten_3:get(1):clearState())
    
        
    local filename_mat = paths.concat(cmd_params.save, 'mmatch.net')
    print('==> saving model to '.. filename_mat)
    torch.save(filename_mat, mmatch:get(1):clearState())
        
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
