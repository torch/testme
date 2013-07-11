require 'torch'
pcall(require,'image')

local tester
local precision

local test = {}

function test.rgb2yuv()
   local rgb = image.lena()

   local yuv = image.rgb2yuv(rgb)
   local rgb2 = image.yuv2rgb(yuv)

   local err = (rgb-rgb2):abs():max()
   
   tester:assertlt(err, precision, 'error in torch.dot')
end

math.randomseed(os.time())

print('')
print('Testing image with type = double')
print('')
torch.setdefaulttensortype('torch.DoubleTensor')
precision = 1e-4
tester = torch.Tester()
tester:add(test)
tester:run()

print('')
print('Testing image with type = float')
print('')
torch.setdefaulttensortype('torch.FloatTensor')
precision = 1e-4
tester = torch.Tester()
tester:add(test)
tester:run()
