require 'torch'

local tester
local precision

local test = {}

function test.dot()
   local v1 = torch.randn(100)
   local v2 = torch.randn(100)

   local res1 = torch.dot(v1,v2)

   local res2 = 0
   for i = 1,v1:size(1) do
      res2 = res2 + v1[i] * v2[i]
   end

   local err = math.abs(res1-res2)
   
   tester:assertlt(err, precision, 'error in torch.dot')
end

function test.mv()
   local m1 = torch.randn(100,100)
   local v1 = torch.randn(100)

   local res1 = torch.mv(m1,v1)

   local res2 = res1:clone():zero()
   for i = 1,m1:size(1) do
      for j = 1,m1:size(2) do
         res2[i] = res2[i] + m1[i][j] * v1[j]
      end
   end

   local err = (res1-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.dot')
end

function test.add()
   local m1 = torch.randn(100,100)
   local v1 = torch.randn(100)

   local res1 = torch.add(m1[{ 4,{} }],v1)

   local res2 = res1:clone():zero()
   for i = 1,m1:size(2) do
      res2[i] = m1[4][i] + v1[i]
   end

   local err = (res1-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.add - contiguous')

   local m1 = torch.randn(100,100)
   local v1 = torch.randn(100)

   local res1 = torch.add(m1[{ {},4 }],v1)

   local res2 = res1:clone():zero()
   for i = 1,m1:size(1) do
      res2[i] = m1[i][4] + v1[i]
   end

   local err = (res1-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.add - non contiguous')

   local m1 = torch.randn(10,10)
   local res1 = m1:clone()

   res1[{ {},3 }]:add(2)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] + 2
   end
   
   local err = (res1-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.add - scalar, non contiguous')
end

function test.mul()
   local m1 = torch.randn(100,100)
   local v1 = torch.randn(100)

   local res1 = torch.cmul(m1[{ 4,{} }],v1)

   local res2 = res1:clone():zero()
   for i = 1,m1:size(2) do
      res2[i] = m1[4][i] * v1[i]
   end

   local err = (res1-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.mul - contiguous')

   local m1 = torch.randn(100,100)
   local v1 = torch.randn(100)

   local res1 = torch.cmul(m1[{ {},4 }],v1)

   local res2 = res1:clone():zero()
   for i = 1,m1:size(1) do
      res2[i] = m1[i][4] * v1[i]
   end

   local err = (res1-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.mul - non contiguous')

   local m1 = torch.randn(10,10)
   local res1 = m1:clone()

   res1[{ {},3 }]:mul(2)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] * 2
   end
   
   local err = (res1-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.mul - scalar, non contiguous')
end

function test.div()
   local m1 = torch.randn(100,100)
   local v1 = torch.rand(100):add(1)

   local res1 = torch.cdiv(m1[{ 4,{} }],v1)

   local res2 = res1:clone():zero()
   for i = 1,m1:size(2) do
      res2[i] = m1[4][i] / v1[i]
   end

   local err = (res1-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.div - contiguous')

   local m1 = torch.randn(100,100)
   local v1 = torch.rand(100):add(1)

   local res1 = torch.cdiv(m1[{ {},4 }],v1)

   local res2 = res1:clone():zero()
   for i = 1,m1:size(1) do
      res2[i] = m1[i][4] / v1[i]
   end

   local err = (res1-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.div - non contiguous')

   local m1 = torch.randn(10,10)
   local res1 = m1:clone()

   res1[{ {},3 }]:div(2)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] / 2
   end
   
   local err = (res1-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.div - scalar, non contiguous')
end

math.randomseed(os.time())

print('')
print('Testing torch with type = double')
print('')
torch.setdefaulttensortype('torch.DoubleTensor')
precision = 1e-8
tester = torch.Tester()
tester:add(test)
tester:run()

print('')
print('Testing torch with type = float')
print('')
torch.setdefaulttensortype('torch.FloatTensor')
precision = 1e-4
tester = torch.Tester()
tester:add(test)
tester:run()
