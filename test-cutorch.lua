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

math.randomseed(os.time())

print('')
print('Testing torch with type = cuda')
print('')
torch.setdefaulttensortype('torch.DoubleTensor')
precision = 1e-5
tester = torch.Tester()
tester:add(test)
tester:run()
