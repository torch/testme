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

function test.abs()  -- [res] torch.abs([res,] x)
   -- contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.abs(m1[{ 4,{} }])
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.abs(m1[4][i])
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
   tester:assertlt(maxerr, precision, 'error in torch.abs - contiguous')
   
   -- non-contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.abs(m1[{ {}, 4 }])
   local res2 = res1:clone():zero()
   for i = 1,res1:size(1) do
      res2[i] = math.abs(m1[i][4])
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
   tester:assertlt(maxerr, precision, 'error in torch.abs - non-contiguous')
end

function test.max()  -- torch.max([resval, resind,] x [,dim])
   -- torch.max( x )
   -- contiguous
   local m1 = torch.randn(100,100)
   local res1 = torch.max(m1)
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
   local m1 = torch.randn(10,10,10)
   local m2 = m1[{{}, 4, {}}]
   local res1 = torch.max(m2)
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
   local m1 = torch.randn(100,100)
   local res1val, res1ind = torch.max(m1, 2)
   local res2val = res1val:clone():zero()
   local res2ind = res1ind:clone():zero()
   for i=1, m1:size(1) do
      res2val[i] = m1[i][1]
      res2ind[i] = 1
      for j=1, m1:size(2) do
	 if m1[i][j] > res2val[i][1] then
	    res2val[i] = m1[i][j]
	    res2ind[i] = j
	 end
      end
   end
   local errval = res1val:clone():zero()
   for i = 1, res1val:size(1) do
      errval[i] = math.abs(res1val[i][1] - res2val[i][1])
      tester:asserteq(res1ind[i][1], res2ind[i][1], 'error in torch.max - non-contiguous')
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
   local m1 = torch.randn(100,100)
   local res1 = torch.min(m1)
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
   local m1 = torch.randn(10,10,10)
   local m2 = m1[{{}, 4, {}}]
   local res1 = torch.min(m2)
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
   local m1 = torch.randn(100,100)
   local res1val, res1ind = torch.min(m1, 2)
   local res2val = res1val:clone():zero()
   local res2ind = res1ind:clone():zero()
   for i=1, m1:size(1) do
      res2val[i] = m1[i][1]
      res2ind[i] = 1
      for j=1, m1:size(2) do
	 if m1[i][j] < res2val[i][1] then
	    res2val[i] = m1[i][j]
	    res2ind[i] = j
	 end
      end
   end
   local errval = res1val:clone():zero()
   for i = 1, res1val:size(1) do
      errval[i] = math.abs(res1val[i][1] - res2val[i][1])
      tester:asserteq(res1ind[i][1], res2ind[i][1], 'error in torch.min - non-contiguous')
   end
   local minerr = 0
   for i = 1, errval:size(1) do
      if errval[i][1] < minerr then
	 minerr = errval[i]
      end
   end
   tester:assertlt(minerr, precision, 'error in torch.min - non-contiguous')      
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
   
   tester:assertlt(err, precision, 'error in torch.mv')
end

function test.add()
   -- [res] torch.add([res,] tensor1, tensor2)
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

   -- [res] torch.add([res,] tensor, value)
   local m1 = torch.randn(10,10)
   local res1 = m1:clone()
   res1[{ 3,{} }]:add(2)
   
   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ 3,i }] = res2[{ 3,i }] + 2
   end
   
   local err = (res1-res2):abs():max()

   tester:assertlt(err, precision, 'error in torch.add - scalar, contiguous')

   local m1 = torch.randn(10,10)
   local res1 = m1:clone()
   res1[{ {},3 }]:add(2)

   local res2 = m1:clone()
   for i = 1,m1:size(1) do
      res2[{ i,3 }] = res2[{ i,3 }] + 2
   end
   
   local err = (res1-res2):abs():max()

   tester:assertlt(err, precision, 'error in torch.add - scalar, non contiguous')
   
   -- [res] torch.add([res,] tensor1, value, tensor2)
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

-- Done. dot, mv, add, mul, div, abs, max, min
-- TODO: cat, diag, eye, linspace, logspace, ones, rand, randn, range, randperm,
--       reshape, tril, triu, zeros, acos, asin, atan, ceil, cos, cosh, exp, floor,
--       log, log1p, pow, sin, sinh, sqrt, tan, tanh, cmul, addcmul, div, cdiv,
--       addcdiv, addmv, addr, addmm, mv, mm, ger, A+B, A-B,-B, A*B, A/x, cross,
--       cumprod, cumsum, mean, prod, sort, std, sum, var, norm, dist, numel,
--       trace, conv2, xcorr2, conv3, xcorr3, gesv, gels, symeig, eig, svd,
--       inverse, lt, le, gt, ge, eq, ne 

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
