require 'torch'
local ok = pcall(require,'cutorch')
if not ok then return end

local tester
local precision

local test = {}

function test.sum()
   local v1 = torch.randn(100):cuda()
   local res1 = v1:sum()
   local res2 = 0
   for i = 1,v1:size(1) do
      res2 = res2 + v1[i]
   end
   local err = math.abs(res1-res2)
   tester:assertlt(err, precision, 'error in torch.sum')
   
   local v1 = torch.randn(3,4,25):cuda()
   local res1 = v1[2]:sum()
   local res2 = 0
   for i = 1,v1[2]:size(1) do
      for j = 1,v1[2]:size(2) do
         res2 = res2 + v1[2][i][j]
      end
   end
   local err = math.abs(res1-res2)
   tester:assertlt(err, precision, 'error in torch.sum')
end

function test.dot()
   local v1 = torch.randn(100):cuda()
   local v2 = torch.randn(100):cuda()

   local res1 = v1.dot(v1,v2)

   local res2 = 0
   for i = 1,v1:size(1) do
      res2 = res2 + v1[i] * v2[i]
   end

   local err = math.abs(res1-res2)
   
   tester:assertlt(err, precision, 'error in torch.dot')
end

local genericSingleOpTest = [[
   -- [res] torch.functionname([res,] x)
   -- contiguous
   local m1 = torch.randn(100,100)
   local res1 = m1:cuda()[{ 4,{} }]
   res1:functionname()
   local res2 = res1:clone():zero():float()
   for i = 1,res1:size(1) do
      res2[i] = math.functionname(m1[4][i])
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerrc = 0
   for i = 1, err:size(1) do
      if err[i] > maxerrc then
	 maxerrc = err[i]
      end
   end      
   
   -- non-contiguous
   local m1 = torch.randn(100,100)
   local res1 = m1:cuda()[{ {}, 4 }]
   res1:functionname()
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.functionname(m1[i][4])
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerrnc = 0
   for i = 1, err:size(1) do
      if err[i] > maxerrnc then
	 maxerrnc = err[i]
      end
   end
   return maxerrc, maxerrnc
]]

function test.abs()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'abs'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.sin()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'sin'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.sinh()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'sinh'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.asin()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'asin'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.cos()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'cos'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.cosh()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'cosh'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.acos()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'acos'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.tan()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'tan'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.tanh()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'tanh'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.atan()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'atan'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.log()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'log'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.sqrt()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'sqrt'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.exp()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'exp'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.floor()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'floor'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.ceil()
   local f = loadstring(string.gsub(genericSingleOpTest, 'functionname', 'ceil'))
   local maxerrc, maxerrnc = f()
   tester:assertlt(maxerrc, precision, 'error in torch.functionname - contiguous')
   tester:assertlt(maxerrnc, precision, 'error in torch.functionname - non-contiguous')
end

function test.max()  -- torch.max([resval, resind,] x [,dim])
   -- torch.max( x )
   -- contiguous
   local m1 = torch.randn(100,100):cuda()
   local res1 = m1:max()
   local res2 = m1[1][1]
   for i = 1,m1:size(1) do
      for j = 1,m1:size(2) do
	 if m1[i][j] > res2 then
	    res2 = m1[i][j]
	 end
      end      
   end
   local err = res1 - res2
   tester:assertlt(err, precision, 'error in torch.max - contiguous')
   -- non-contiguous
   local m1 = torch.randn(10,10,10):cuda()
   local m2 = m1[{{}, 4, {}}]
   local res1 = m2:max()
   local res2 = m2[1][1]
   for i = 1,m2:size(1) do
      for j = 1,m2:size(2) do
	 if m2[i][j] > res2 then
	    res2 = m2[i][j]
	 end
      end      
   end
   local err = res1 - res2
   tester:assertlt(err, precision, 'error in torch.max - non-contiguous')
   -- torch.max([resval, resind,] x ,dim])
   local m1 = torch.randn(100,100):cuda()
   local res1val, res1ind = m1:max(2)
   local res2val = res1val:clone():zero()
   -- TODO: Disabling indices check due to BUG. See CUDA Caveats section below 
   -- local res2ind = res1ind:clone():zero()
   for i=1, m1:size(1) do
      res2val[i] = m1[i][1]
      -- res2ind[i] = 1
      for j=1, m1:size(2) do
	 if m1[i][j] > res2val[i][1] then
	    res2val[i] = m1[i][j]
	    -- res2ind[i] = j
	 end
      end
   end
   local errval = res1val:clone():zero()
   for i = 1, res1val:size(1) do
      errval[i] = math.abs(res1val[i][1] - res2val[i][1])
      -- tester:asserteq(res1ind[i][1], res2ind[i][1], 'error in torch.max - non-contiguous')
   end
   local maxerr = 0
   for i = 1, errval:size(1) do
      if errval[i][1] > maxerr then
	 maxerr = errval[i]
      end
   end
   tester:assertlt(maxerr, precision, 'error in torch.max - non-contiguous')      
end

function test.min()  -- torch.min([resval, resind,] x [,dim])
   -- torch.min( x )
   -- contiguous
   local m1 = torch.randn(100,100):cuda()
   local res1 = m1:min()
   local res2 = m1[1][1]
   for i = 1,m1:size(1) do
      for j = 1,m1:size(2) do
	 if m1[i][j] < res2 then
	    res2 = m1[i][j]
	 end
      end      
   end
   local err = res1 - res2
   tester:assertlt(err, precision, 'error in torch.min - contiguous')
   -- non-contiguous
   local m1 = torch.randn(10,10,10):cuda()
   local m2 = m1[{{}, 4, {}}]
   local res1 = m2:min()
   local res2 = m2[1][1]
   for i = 1,m2:size(1) do
      for j = 1,m2:size(2) do
	 if m2[i][j] < res2 then
	    res2 = m2[i][j]
	 end
      end      
   end
   local err = res1 - res2
   tester:assertlt(err, precision, 'error in torch.min - non-contiguous')
   -- torch.min([resval, resind,] x ,dim])
   local m1 = torch.randn(100,100):cuda()
   local res1val, res1ind = m1:min(2)
   local res2val = res1val:clone():zero()
   -- TODO: Disabling indices check due to BUG. See CUDA Caveats section below 
   -- local res2ind = res1ind:clone():zero()
   for i=1, m1:size(1) do
      res2val[i] = m1[i][1]
      -- res2ind[i] = 1
      for j=1, m1:size(2) do
	 if m1[i][j] < res2val[i][1] then
	    res2val[i] = m1[i][j]
	    -- res2ind[i] = j
	 end
      end
   end
   local errval = res1val:clone():zero()
   for i = 1, res1val:size(1) do
      errval[i] = math.abs(res1val[i][1] - res2val[i][1])
      -- tester:asserteq(res1ind[i][1], res2ind[i][1], 'error in torch.min - non-contiguous')
   end
   local minerr = 0
   for i = 1, errval:size(1) do
      if errval[i][1] < minerr then
	 minerr = errval[i]
      end
   end
   tester:assertlt(minerr, precision, 'error in torch.min - non-contiguous')      
end

function test.add()
   -- [res] torch.add([res,] tensor1, tensor2)
   local m1 = torch.randn(100,100)   
   local v1 = torch.randn(100)
   local m1c = m1:cuda()
   local v1c = v1:cuda()

   local res1 = m1c[{ 4,{} }]
   res1:add(v1c)

   local res2 = res1:float():clone():zero()
   for i = 1,m1:size(2) do
      res2[i] = m1[4][i] + v1[i]
   end

   local err = (res1:float()-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.add - contiguous')

   local m1 = torch.randn(100,100)
   local v1 = torch.randn(100)
   local m1c = m1:cuda()
   local v1c = v1:cuda()

   local res1 = m1c[{ {},4 }]
   res1:add(v1c)

   local res2 = res1:float():clone():zero()
   for i = 1,m1:size(1) do
      res2[i] = m1[i][4] + v1[i]
   end

   local err = (res1:float()-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.add - non contiguous')

   -- [res] torch.add([res,] tensor, value)
   local m1 = torch.randn(10,10)
   local res1 = m1:cuda()
   res1[{ 3,{} }]:add(2)
   
   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ 3,i }] = res2[{ 3,i }] + 2
   end
   local err = (res1:float()-res2):abs():max()

   tester:assertlt(err, precision, 'error in torch.add - scalar, contiguous')

   local m1 = torch.randn(10,10)
   local res1 = m1:cuda()
   res1[{ {},3 }]:add(2)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] + 2
   end
   
   local err = (res1:float()-res2):abs():max()

   tester:assertlt(err, precision, 'error in torch.add - scalar, non contiguous')
   
   -- [res] torch.add([res,] tensor1, value, tensor2)
end

function test.mul()
   local m1 = torch.randn(10,10)
   local res1 = m1:clone():cuda()

   res1[{ {},3 }]:mul(2)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] * 2
   end
   
   local err = (res1:float()-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.mul - scalar, non contiguous')
end

function test.div()
   local m1 = torch.randn(10,10)
   local res1 = m1:clone():cuda()

   res1[{ {},3 }]:div(2)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] / 2
   end
   
   local err = (res1:float()-res2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.div - scalar, non contiguous')
end

function test.pow()  -- [res] torch.pow([res,] x)
   -- contiguous
   local m1 = torch.randn(100,100)
   local m1c = m1:cuda()
   local res1 = m1c[{ 4,{} }]
   res1:pow(3)
   local res2 = res1:clone():zero():float()
   for i = 1,res1:size(1) do
      res2[i] = math.pow(m1[4][i], 3)
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      if err[i] > maxerr then
	 maxerr = err[i]
      end
   end   
   tester:assertlt(maxerr, precision, 'error in torch.pow - contiguous')
   
   -- non-contiguous
   local m1 = torch.randn(100,100)
   local m1c = m1:cuda()
   local res1 = m1c[{ {}, 4 }]
   res1:pow(3)
   local res2 = res1:clone():zero():float()
   for i = 1,res1:size(1) do
      res2[i] = math.pow(m1[i][4], 3)
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      err[i] = math.abs(res1[i] - res2[i])
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      if err[i] > maxerr then
	 maxerr = err[i]
      end
   end   
   tester:assertlt(maxerr, precision, 'error in torch.pow - non-contiguous')
end

function test.cdiv()  -- [res] torch.cdiv([res,] tensor1, tensor2)
   -- contiguous
   local m1 = torch.randn(10, 10, 10):cuda()
   local m2 = torch.randn(10, 10 * 10):cuda()
   local res1 = m1[{4, {}, {}}]
   local sm1 = res1:float()
   local sm2 = m2[{4, {}}]
   res1:cdiv(sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
	 local idx1d = (((i-1)*sm1:size(1)))+j 
	 res2[i][j] = sm1[i][j] / sm2[idx1d]
      end
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      for j = 1, res1:size(2) do
	 err[i][j] = math.abs(res1[i][j] - res2[i][j])
      end
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      for j = 1, err:size(2) do
	 if err[i][j] > maxerr then
	    maxerr = err[i][j]
	 end
      end
   end   
   tester:assertlt(maxerr, precision, 'error in torch.cdiv - contiguous')

   -- non-contiguous
   local m1 = torch.randn(10, 10, 10):cuda()
   local m2 = torch.randn(10 * 10, 10 * 10):cuda()  
   local res1 = m1[{{}, 4, {}}]
   local sm1 = res1:clone()
   local sm2 = m2[{{}, 4}]
   res1:cdiv(sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
	 local idx1d = (((i-1)*sm1:size(1)))+j 
	 res2[i][j] = sm1[i][j] / sm2[idx1d]
      end
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      for j = 1, res1:size(2) do
	 err[i][j] = math.abs(res1[i][j] - res2[i][j])
      end
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      for j = 1, err:size(2) do
	 if err[i][j] > maxerr then
	    maxerr = err[i][j]
	 end
      end
   end   
   tester:assertlt(maxerr, precision, 'error in torch.cdiv - non-contiguous')  
end

function test.cmul()  -- [res] torch.cmul([res,] tensor1, tensor2)
   -- contiguous
   local m1 = torch.randn(10, 10, 10):cuda()
   local m2 = torch.randn(10, 10 * 10):cuda()
   local res1 = m1[{4, {}, {}}]
   local sm1 = res1:float()
   local sm2 = m2[{4, {}}]
   res1:cmul(sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
	 local idx1d = (((i-1)*sm1:size(1)))+j 
	 res2[i][j] = sm1[i][j] * sm2[idx1d]
      end
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      for j = 1, res1:size(2) do
	 err[i][j] = math.abs(res1[i][j] - res2[i][j])
      end
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      for j = 1, err:size(2) do
	 if err[i][j] > maxerr then
	    maxerr = err[i][j]
	 end
      end
   end   
   tester:assertlt(maxerr, precision, 'error in torch.cmul - contiguous')

   -- non-contiguous
   local m1 = torch.randn(10, 10, 10):cuda()
   local m2 = torch.randn(10 * 10, 10 * 10):cuda()
   local res1 = m1[{{}, 4, {}}]
   local sm1 = res1:float()
   local sm2 = m2[{{}, 4}]
   res1:cmul(sm2)
   local res2 = res1:clone():zero()
   for i = 1,sm1:size(1) do
      for j = 1, sm1:size(2) do
	 local idx1d = (((i-1)*sm1:size(1)))+j 
	 res2[i][j] = sm1[i][j] * sm2[idx1d]
      end
   end
   local err = res1:clone():zero()
   -- find absolute error
   for i = 1, res1:size(1) do
      for j = 1, res1:size(2) do
	 err[i][j] = math.abs(res1[i][j] - res2[i][j])
      end
   end
   -- find maximum element of error
   local maxerr = 0
   for i = 1, err:size(1) do
      for j = 1, err:size(2) do
	 if err[i][j] > maxerr then
	    maxerr = err[i][j]
	 end
      end
   end   
   tester:assertlt(maxerr, precision, 'error in torch.cmul - non-contiguous')  
end

-- CUDA caveats -- 
--[[
   where t is a torch.CudaTensor
   You cannot do torch.sin(t) 
   You can only do t:sin()

   max and min functions on a particular dimension do not return the index of the max element, 
   but just the max value. (this is not the definition according to the manual)

   torch.mv is not implemented
   torch.ones is not implemented
   torch.zeros is not implemented

--]]
-- Done. dot, add, mul, div, abs, max, min, sin, sinh, cos, cosh, 
--       tan, tanh, asin, acos, atan, log, sqrt, exp, floor, ceil,
--       cdiv, cmul, pow,
-- TODO: cat, diag, eye, linspace, 
--       logspace, rand, randn, range, randperm,
--       reshape, tril, triu, log1p, addcmul, lt, le, gt, ge, eq, ne
--       addcdiv, addmv, addr, addmm, mm, ger, A+B, A-B,-B, A*B, A/x, cross,
--       cumprod, cumsum, mean, prod, sort, std, sum, var, norm, dist, numel,
--       trace, conv2, xcorr2, conv3, xcorr3, gesv, gels, symeig, eig, svd,
--       inverse
--       Fundamental tensor functions from here: http://www.torch.ch/manual/torch/tensor

math.randomseed(os.time())

print('')
print('Testing torch with type = cuda')
print('')
torch.setdefaulttensortype('torch.FloatTensor')
precision = 1e-4
tester = torch.Tester()
tester:add(test)
tester:run()
