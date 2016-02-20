require './init'

local data = torch.rand(5, 3):split(1)

print(data)

local iterator = zd.Iterator {
   source = data ,
   batch = 3
}

print( iterator:batchSize() )

iterator:reset()
repeat
    local entry = iterator:next()
    print( entry )
until iterator:finished()

iterator:each( print )

