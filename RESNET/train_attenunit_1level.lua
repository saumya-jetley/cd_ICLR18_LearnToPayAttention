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
  model = '1_resnet_local',
  model_2 = '2_resnet_global',
  model_3 = '3_atten',
  model_4 = '4_match_singleimagepred',
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

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
local model_2 = nn.Sequential()
local model_3 = nn.Sequential()
local model_4 = nn.Sequential()

local net = dofile('models/Resnet-att1/'..opt.model..'.lua'):cuda()
local net_2 = dofile('models/Resnet-att1/'..opt.model_2..'.lua'):cuda()
local net_3 = dofile('models/Resnet-att1/'..opt.model_3..'.lua'):cuda()
local net_4 = dofile('models/Resnet-att1/'..opt.model_4..'.lua'):cuda()

do
   ------Main Model------
   local function add(flag, module) if flag then model:add(module) end end   ----function definition
   add(opt.hflip, nn.BatchFlip():float())
   add(opt.randomcrop > 0, nn.RandomCrop(opt.randomcrop, opt.randomcrop_type):float())
   model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
   add(opt.multiply_input_factor ~= 1, nn.MulConstant(opt.multiply_input_factor):cuda())
   
   model:add(net) ---- adding the network -- sjmod

   cudnn.convert(net, cudnn)
   cudnn.benchmark = true
   if opt.cudnn_fastest then
      for i,v in ipairs(net:findModules'cudnn.SpatialConvolution') do v:fastest() end
   end
   if opt.cudnn_deterministic then
      model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
   end

   print(net)
   print('Network has', #model:findModules'cudnn.SpatialConvolution', 'convolutions')

   local sample_input = torch.randn(8,3,opt.imageSize,opt.imageSize):cuda()
   if opt.generate_graph then
      iterm.dot(graphgen(net, sample_input), opt.save..'/graph.pdf')
   end
   if opt.optnet_optimize then
      optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
   end
    
   ------Global, Atten and Match Models------
   model_2:add(net_2) ---- adding the network -- sjmod
   model_3:add(net_3) ---- adding the network -- sjmod
   model_4:add(net_4) ---- adding the network -- sjmod
   
   cudnn.convert(net_2, cudnn)
   cudnn.convert(net_3, cudnn)
   cudnn.convert(net_4, cudnn)
   
   cudnn.benchmark = true
   if opt.cudnn_fastest then
      for i,v in ipairs(net_2:findModules'cudnn.SpatialConvolution') do v:fastest() end
      for i,v in ipairs(net_3:findModules'cudnn.SpatialConvolution') do v:fastest() end
      for i,v in ipairs(net_4:findModules'cudnn.SpatialConvolution') do v:fastest() end
   end
    --[[
    if opt.cudnn_deterministic then
      model_2:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
      model_3:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
   end
    ]]--

   print(net_2)
   print('Network has', #model_2:findModules'cudnn.SpatialConvolution', 'convolutions')

   print(net_3)
   print('Network has', #model_3:findModules'cudnn.SpatialConvolution', 'convolutions')
    
   print(net_4)
   print('Network has', #model_4:findModules'cudnn.SpatialConvolution', 'convolutions')
    
end

local function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end

print('Will save at '..opt.save)
paths.mkdir(opt.save)

model_all = {}
table.insert(model_all, model)
table.insert(model_all, model_2)
table.insert(model_all, model_3)
table.insert(model_all, model_4)
local parameters, gradParameters = model_utils.combine_all_parameters(model_all)
--local parameters,gradParameters = model:getParameters()   --sjmod

opt.n_parameters = parameters:numel()
print('Network has ', parameters:numel(), 'parameters')

print(c.blue'==>' ..' setting criterion')
local criterion = nn.CrossEntropyCriterion():cuda()

-- a-la autograd
local f = function(inputs, targets)
   local lfeat = model:forward(inputs)   ----sjmod
   local gfeat = model_2:forward(lfeat)
   local att_con = model_3:forward({lfeat,gfeat})
    local pred = model_4:forward(att_con[2])

   local loss = criterion:forward(pred, targets)
   local df_pred = criterion:backward(pred, targets)
    
   local df_att_con2 = model_4:backward(att_con[2], df_pred)  ----sjmod
   local df_feat = model_3:backward({lfeat,gfeat}, {torch.rand(att_con[1]:size()):cuda():fill(0), df_att_con2})  ----sjmod
   local df_lfeat = model_2:backward(lfeat, df_feat[2])
   model:backward(inputs, (df_lfeat+df_feat[1])/2)

   return loss
end

print(c.blue'==>' ..' configuring optimizer')
local optimState = tablex.deepcopy(opt)


function train()
  model:training()

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
  model:evaluate()
  local confusion = optim.ConfusionMatrix(opt.num_classes)
  local data_split = provider.testData.data:split(opt.batchSize,1)
  local labels_split = provider.testData.labels:split(opt.batchSize,1)

  for i,v in ipairs(data_split) do
      local lfeat = model:forward(v)   ----sjmod
      local gfeat = model_2:forward(lfeat)
      local att_con = model_3:forward({lfeat,gfeat})
      local pred = model_4:forward(att_con[2])
      
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
    torch.save(opt.save..'/model.t7', net:clearState())	
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

torch.save(opt.save..'/model.t7', net:clearState())
