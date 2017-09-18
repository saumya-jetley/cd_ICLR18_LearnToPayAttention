-- Code for Wide Residual Networks http://arxiv.org/abs/1605.07146
-- (c) Sergey Zagoruyko, 2016
require 'xlua'
require 'optim'
require 'image'
require 'cunn'
require 'cudnn'
local c = require 'trepl.colorize'
local json = require 'cjson'
paths.dofile'augmentation.lua'
model_utils = require 'model_utils'   ---sjmod

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
local iterm = require 'iterm'
require 'iterm.dot'

opt = {
  dataset = './datasets/cifar10_whitened.t7',
  save = 'logs',
  batchSize = 64, --128,
  learningRate = 0.1,   ------overwritten
  learningRateDecay = 0,   ------overwritten
  learningRateDecayRatio = 0.2,   ------overwritten
  weightDecay = 0.0005,
  dampening = 0,
  momentum = 0.9,
  epoch_step = "80",   ------overwritten
  max_epoch = 300,   ------overwritten

  model_1_1 = '1.1_resnet_local',

  model_1_2 = '1.2_resnet_local',
  model_3_2 = '3.2_atten',

  model_1_3 = '1.3_resnet_local',
  model_2_3 = '2.3_resnet_global',
  model_3_3 = '3.3_atten',
  
  model_4 = '4_match_singleimagepred_indep',

  optimMethod = 'sgd',
  init_value = 10,
  depth = 50,
  shortcutType = 'A',
  nesterov = false,   ------overwritten
  dropout = 0,
  hflip = true,
  randomcrop = 4,
  imageSize = 32,
  randomcrop_type = 'zero',   ------overwritten
  cudnn_fastest = true,
  cudnn_deterministic = false,
  optnet_optimize = true,
  generate_graph = false,
  multiply_input_factor = 1,
  widen_factor = 1,
}
opt = xlua.envparams(opt)

opt.epoch_step = tonumber(opt.epoch_step) or loadstring('return '..opt.epoch_step)()
print(opt)

print(c.blue '==>' ..' loading data')
local provider = torch.load(opt.dataset)
opt.num_classes = provider.testData.labels:max()

print(c.blue '==>' ..' configuring models')
----------------------------------------------------------------
local model_1_1 = nn.Sequential()
local model_1_2 = nn.Sequential()
local model_1_3 = nn.Sequential()

local model_2_3 = nn.Sequential()

local model_3_2 = nn.Sequential()
local model_3_3 = nn.Sequential()

local model_4 = nn.Sequential()
----------------------------------------------------------------

local net_1_1 = dofile('models/Resnet-att2/'..opt.model_1_1..'.lua'):cuda()

local net_1_2 = dofile('models/Resnet-att2/'..opt.model_1_2..'.lua'):cuda()
local net_3_2 = dofile('models/Resnet-att2/'..opt.model_3_2..'.lua'):cuda()

local net_1_3 = dofile('models/Resnet-att2/'..opt.model_1_3..'.lua'):cuda()
local net_2_3 = dofile('models/Resnet-att2/'..opt.model_2_3..'.lua'):cuda()
local net_3_3 = dofile('models/Resnet-att2/'..opt.model_3_3..'.lua'):cuda()

local net_4 = dofile('models/Resnet-att2/'..opt.model_4..'.lua'):cuda()
----------------------------------------------------------------

do
   ------Main Model----------INITIALIZATION-----------------
   local function add(flag, module) if flag then model_1_1:add(module) end end   ----function definition
   add(opt.hflip, nn.BatchFlip():float())
   add(opt.randomcrop > 0, nn.RandomCrop(opt.randomcrop, opt.randomcrop_type):float())
   model_1_1:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
   add(opt.multiply_input_factor ~= 1, nn.MulConstant(opt.multiply_input_factor):cuda())
   model_1_1:add(net_1_1) ---- adding the network -- sjmod

   cudnn.convert(net_1_1, cudnn)
   cudnn.benchmark = true
   if opt.cudnn_fastest then
      for i,v in ipairs(net_1_1:findModules'cudnn.SpatialConvolution') do v:fastest() end
   end
   if opt.cudnn_deterministic then
      model_1_1:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
   end

   print(net_1_1)
   print('Network has', #model_1_1:findModules'cudnn.SpatialConvolution', 'convolutions')

   local sample_input = torch.randn(8,3,opt.imageSize,opt.imageSize):cuda()
   if opt.generate_graph then
      iterm.dot(graphgen(net_1_1, sample_input), opt.save..'/graph.pdf')
   end
   if opt.optnet_optimize then
      optnet.optimizeMemory(net_1_1, sample_input, {inplace = false, mode = 'training'})
   end
    
 
  ------Global, Atten and Match Models----------INITIALIZATION-----------------

   model_1_2:add(net_1_2) ---- adding the network -- sjmod
   model_3_2:add(net_3_2) ---- adding the network -- sjmod

   model_1_3:add(net_1_3) ---- adding the network -- sjmod
   model_2_3:add(net_2_3) ---- adding the network -- sjmod
   model_3_3:add(net_3_3) ---- adding the network -- sjmod

   model_4:add(net_4) ---- adding the network -- sjmod
   
   ------
   cudnn.convert(net_1_2, cudnn)
   cudnn.convert(net_3_2, cudnn)

   cudnn.convert(net_1_3, cudnn)
   cudnn.convert(net_2_3, cudnn)
   cudnn.convert(net_3_3, cudnn)

   cudnn.convert(net_4, cudnn)

   ------
   cudnn.benchmark = true
   if opt.cudnn_fastest then   
      for i,v in ipairs(net_1_2:findModules'cudnn.SpatialConvolution') do v:fastest() end
      for i,v in ipairs(net_3_2:findModules'cudnn.SpatialConvolution') do v:fastest() end
      
      for i,v in ipairs(net_1_3:findModules'cudnn.SpatialConvolution') do v:fastest() end
      for i,v in ipairs(net_2_3:findModules'cudnn.SpatialConvolution') do v:fastest() end
      for i,v in ipairs(net_3_3:findModules'cudnn.SpatialConvolution') do v:fastest() end
      
      for i,v in ipairs(net_4:findModules'cudnn.SpatialConvolution') do v:fastest() end
   end
    --[[
    if opt.cudnn_deterministic then
      model_2:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
      model_3:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
   end
    ]]--
   ------
   print(net_1_2); print('Network has', #model_1_2:findModules'cudnn.SpatialConvolution', 'convolutions')
   print(net_3_2); print('Network has', #model_3_2:findModules'cudnn.SpatialConvolution', 'convolutions')

   print(net_1_3); print('Network has', #model_1_3:findModules'cudnn.SpatialConvolution', 'convolutions')
   print(net_2_3); print('Network has', #model_2_3:findModules'cudnn.SpatialConvolution', 'convolutions')
   print(net_3_3); print('Network has', #model_3_3:findModules'cudnn.SpatialConvolution', 'convolutions')

   print(net_4); print('Network has', #model_4:findModules'cudnn.SpatialConvolution', 'convolutions')
    
end


local function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end

print('Will save at '..opt.save)
paths.mkdir(opt.save)

model_all = {}
table.insert(model_all, model_1_1)

table.insert(model_all, model_1_2)
table.insert(model_all, model_3_2)

table.insert(model_all, model_1_3)
table.insert(model_all, model_2_3)
table.insert(model_all, model_3_3)

table.insert(model_all, model_4)
local parameters, gradParameters = model_utils.combine_all_parameters(model_all)
--local parameters,gradParameters = model:getParameters()   --sjmod

opt.n_parameters = parameters:numel()
print('Network has ', parameters:numel(), 'parameters')

print(c.blue'==>' ..' setting criterion')
local criterion = nn.CrossEntropyCriterion():cuda()




-- a-la autograd
local f = function(inputs, targets)
------Forward
   local lfeat_160 = model_1_1:forward(inputs)
   local lfeat_320 = model_1_2:forward(lfeat_160)
   local lfeat_640 = model_1_3:forward(lfeat_320)

   local gfeat_640 = model_2_3:forward(lfeat_640)

   local att_con_320 = model_3_2:forward({lfeat_320,gfeat_640})
   local att_con_640 = model_3_3:forward({lfeat_640,gfeat_640})

   local pred = model_4:forward({att_con_320[2],att_con_640[2]})
------Loss
   local loss = criterion:forward(pred, targets)
   local df_pred = criterion:backward(pred, targets)
------Backward
   local df_att_con_combined2 = model_4:backward({att_con_320[2],att_con_640[2]}, df_pred)

   local df_feat_640 = model_3_3:backward({lfeat_640,gfeat_640}, {torch.rand(att_con_640[1]:size()):cuda():fill(0), df_att_con_combined2[2]})
   local df_feat_320 = model_3_2:backward({lfeat_320,gfeat_640}, {torch.rand(att_con_320[1]:size()):cuda():fill(0), df_att_con_combined2[1]})
   
   local df_lfeat_640 = model_2_3:backward(lfeat_640,(df_feat_640[2]+df_feat_320[2])/2)

   local df_lfeat_chain_320 = model_1_3:backward(lfeat_320, (df_lfeat_640+df_feat_640[1])/2)
   local df_lfeat_chain_160 = model_1_2:backward(lfeat_160, (df_feat_320[1]+df_lfeat_chain_320)/2)
   model_1_1:backward(inputs, df_lfeat_chain_160)

   return loss
end



print(c.blue'==>' ..' configuring optimizer')
local optimState = tablex.deepcopy(opt)




function train()
  model_1_1:training()

  model_1_2:training()
  model_3_2:training()

  model_1_3:training()
  model_2_3:training()
  model_3_3:training()

  model_4:training()

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all minibatches have equal size
  indices[#indices] = nil

  local loss = 0

  for t,v in ipairs(indices) do
    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    optim[opt.optimMethod](function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      loss = loss + f(inputs, targets)    ------training fwd and bwd  ---sjmod
      return f,gradParameters
    end, parameters, optimState)
  end

  return loss / #indices
end



function test()
  model_1_1:evaluate()

  model_1_2:evaluate()
  model_3_2:evaluate()

  model_1_3:evaluate()
  model_2_3:evaluate()
  model_3_3:evaluate()

  model_4:evaluate()


  local confusion = optim.ConfusionMatrix(opt.num_classes)
  local data_split = provider.testData.data:split(opt.batchSize,1)
  local labels_split = provider.testData.labels:split(opt.batchSize,1)

  for i,v in ipairs(data_split) do
   local lfeat_160 = model_1_1:forward(v)
   local lfeat_320 = model_1_2:forward(lfeat_160)
   local lfeat_640 = model_1_3:forward(lfeat_320)

   local gfeat_640 = model_2_3:forward(lfeat_640)

   local att_con_320 = model_3_2:forward({lfeat_320,gfeat_640})
   local att_con_640 = model_3_3:forward({lfeat_640,gfeat_640})

   local pred = model_4:forward({att_con_320[2],att_con_640[2]})
      
   confusion:batchAdd(pred, labels_split[i])   -----testing fwd   ----sjmod
  end

  confusion:updateValids()
  return confusion.totalValid * 100
end


for epoch=1,opt.max_epoch do

  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  -- drop learning rate and reset momentum vector
  if torch.type(opt.epoch_step) == 'number' and epoch % opt.epoch_step == 0 or
     torch.type(opt.epoch_step) == 'table' and tablex.find(opt.epoch_step, epoch) then
    opt.learningRate = opt.learningRate * opt.learningRateDecayRatio
    optimState = tablex.deepcopy(opt)
------------save the models-------------------------------------
  torch.save(opt.save..'/model_1_1.t7', net_1_1:clearState())	

  torch.save(opt.save..'/model_1_2.t7', net_1_2:clearState())	
  torch.save(opt.save..'/model_3_2.t7', net_3_2:clearState())	

  torch.save(opt.save..'/model_1_3.t7', net_1_3:clearState())	
  torch.save(opt.save..'/model_2_3.t7', net_2_3:clearState())	
  torch.save(opt.save..'/model_3_3.t7', net_3_3:clearState())	
  
  torch.save(opt.save..'/model_4.t7', net_4:clearState())	
----------------------------------------------------------------
  end

  local function t(f) local s = torch.Timer(); return f(), s:time().real end

  local loss, train_time = t(train)   ----sjmod
  print('training done')
  local test_acc, test_time = t(test)  ----sjmod

  log{
     loss = loss,
     epoch = epoch,
     test_acc = test_acc,
     lr = opt.learningRate,
     train_time = train_time,
     test_time = test_time,
   }
end

------------save the models-------------------------------------
  torch.save(opt.save..'/model_1_1.t7', net_1_1:clearState())	

  torch.save(opt.save..'/model_1_2.t7', net_1_2:clearState())	
  torch.save(opt.save..'/model_3_2.t7', net_3_2:clearState())	

  torch.save(opt.save..'/model_1_3.t7', net_1_3:clearState())	
  torch.save(opt.save..'/model_2_3.t7', net_2_3:clearState())	
  torch.save(opt.save..'/model_3_3.t7', net_3_3:clearState())	
  
  torch.save(opt.save..'/model_4.t7', net_4:clearState())	
----------------------------------------------------------------
