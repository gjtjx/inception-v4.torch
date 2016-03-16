require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local net = nn.Sequential()

local function Tower(layers)
  local tower = nn.Sequential()
  for i=1,#layers do
    tower:add(layers[i])
  end
  return tower
end

local function FilterConcat(towers)
  local concat = nn.DepthConcat(2)
  for i=1,#towers do
    concat:add(towers[i])
  end
  return concat
end

local function Stem()
  local stem = nn.Sequential()
  stem:add(nn.SpatialConvolution(3, 32, 3, 3, 2, 2))
  stem:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1))
  stem:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
  stem:add(Concat(
    {
      nn.SpatialMaxPooling(3, 3, 2, 2),
      nn.SpatialConvolution(64, 96, 3, 3, 2, 2)
    }
  ))
  stem:add(Concat(
    {
      Tower(
        {
          nn.SpatialConvolution(160, 64, 1, 1, 1, 1),
          nn.SpatialConvolution(64, 96, 3, 3, 1, 1)
        }
      ),
      Tower(
        {
          nn.SpatialConvolution(160, 64, 1, 1, 1, 1),
          nn.SpatialConvolution(64, 64, 7, 1, 1, 1),
          nn.SpatialConvolution(64, 64, 1, 7, 1, 1),
          nn.SpatialConvolution(64, 96, 3, 3, 1, 1)
        }
      )
    }
  ))
  stem:add(Concat(
    {
      nn.SpatialConvolution(192, 192, 3, 3, 1, 1),
      nn.SpatialMaxPooling(3, 3, 2, 2)
    }
  ))
  return stem
end
