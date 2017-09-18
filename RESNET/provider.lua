require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class 'Provider'

function Provider:__init(full)
  

  local trsize = 50000
  local tesize = 100

 ---------train set-------------------
   -- download dataset
  if not paths.dirp('cifar-10-batches-t7') then
     local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
     local tar = paths.basename(www)
     os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
  end

  -- load dataset
  self.trainData = {
     data = torch.Tensor(50000, 3072),
     labels = torch.Tensor(50000),
     size = function() return trsize end
  }
  local trainData = self.trainData
  for i = 0,4 do
     local subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
     trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
     trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
  end
  trainData.labels = trainData.labels + 1


 --------test set---------------------  
local subset1 = torch.load('t7-files/airplane.t7')
local subset2 = torch.load('t7-files/automobile.t7')
local subset3 = torch.load('t7-files/birds.t7')
local subset4 = torch.load('t7-files/cats.t7')
local subset5 = torch.load('t7-files/deer.t7')
local subset6 = torch.load('t7-files/dog.t7')
local subset7 = torch.load('t7-files/frog.t7')
local subset8 = torch.load('t7-files/horse.t7')
local subset9 = torch.load('t7-files/ship.t7')
local subset10 = torch.load('t7-files/truck.t7')

  self.testData = {
    data = torch.cat({subset1.data:double(),subset2.data:double(),subset3.data:double(), 
        subset4.data:double(),subset5.data:double(),subset6.data:double(), 
        subset7.data:double(),subset8.data:double(),subset9.data:double(), 
        subset10.data:double()},1),
    labels = torch.cat({subset1.label:double(),subset2.label:double(),subset3.label:double(),
        subset4.label:double(),subset5.label:double(),subset6.label:double(),
        subset7.label:double(),subset8.label:double(),subset9.label:double(),
        subset10.label:double()},1),
    size = function() return tesize end
  }
  local testData = self.testData

  -- resize dataset (if using small version)
  trainData.data = trainData.data[{ {1,trsize} }]
  trainData.labels = trainData.labels[{ {1,trsize} }]

  testData.data = testData.data[{ {1,tesize} }]
  testData.labels = testData.labels[{ {1,tesize} }]
  -- reshape data
  trainData.data = trainData.data:reshape(trsize,3,32,32)
  testData.data = testData.data:reshape(tesize,3,32,32)

end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  local trainData = self.trainData
  local testData = self.testData

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()

  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end

    -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = mean_u
  trainData.std_u = std_u
  trainData.mean_v = mean_v
  trainData.std_v = std_v

  -- preprocess testSet
  for i = 1,testData:size() do
    xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)
end
