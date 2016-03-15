#!/usr/bin/env th


require 'nn'
require 'zd'

local test = torch.TestSuite()

local tester = torch.Tester()

test['JoinTable (from torch)'] = function()
   local tensor = torch.rand(3,4,5)
   local input = {tensor, tensor}
   local module
   for d = 1,tensor:dim() do
      module = zdnn.JoinTable(d)
      tester:asserteq(module:forward(input):size(d), tensor:size(d)*2, "dimension " .. d)
   end

   -- Minibatch
   local tensor = torch.rand(3,4,5)
   local input = {tensor, tensor}
   local module
   for d = 1,tensor:dim()-1 do
      module = zdnn.JoinTable(d, 2)
      tester:asserteq(module:forward(input):size(d+1), tensor:size(d+1)*2, "dimension " .. d)
   end
end

test['JoinTable: empty gradOutput'] = function()
   local tensor = torch.rand(3,4,5)
   local module = zdnn.JoinTable(1)
   local empty = torch.Tensor()
   
   local input = { tensor, tensor, tensor }
   local out = module:forward(input)

   local gi
   gi = module:backward( input, nil )

   tester:asserteq( #gi, 0, "gradInput should be {}")

   gi = module:backward( input, empty )
   for i = 1,#input do
       tester:assertTensorEq( gi[i] , empty, 0,
          "gradInput[" .. i .. "] shoule be an empty tensor")
   end
end

test['JoinTable: same type'] = function()
   local tensor = torch.rand(3,4,5)
   local module

   local types = { 
        "torch.LongTensor", -- "torch.FloatTensor", 
        -- "torch.IntTensor", "torch.DoubleTensor" 
   }

   for _,Type in ipairs(types) do
       tensor = tensor:type(Type)

       local input = {tensor, tensor}

       for d = 1,tensor:dim() do
          module = zdnn.JoinTable(d)
          local out = module:forward(input)
          local sz = tensor:size(d)

          tester:asserteq(out:type(), tensor:type(), "type different")
          tester:asserteq(out:size(d), sz*2, "dimension " .. d)
          tester:assertTensorEq(out:narrow(d, 1,sz), tensor, 1e-7, "value error")
          tester:assertTensorEq(out:narrow(d,sz,sz), tensor, 1e-7, "value error")
       end
   end

   -- Minibatch
   local tensor = torch.rand(3,4,5)
   local module

   for _,Type in ipairs(types) do
       tensor = tensor:type(Type)
       local input = {tensor, tensor}

       for d = 1,tensor:dim()-1 do
          module = zdnn.JoinTable(d, 2)
          local out = module:forward(input)
          local sz = tensor:size(d)

          tester:asserteq(out:type(), tensor:type(), "type different")
          tester:asserteq(out:size(d+1), tensor:size(d+1)*2, "dimension " .. d)
       end
   end
end


function test.FrozenLookupTable()
    local dim = 40
    local db = torch.rand( 30, dim )
    
    local indices_list = {
        {1,21,30,24,16,2,1,9,10}, 
        {25,12,7,24,5,8,6,19,15,1},
        {6,2,11,30,29,3,14,8,8,23,25,17,15,19,25,14,3,4,17,14,20,23,6,18,16,24,10,10,19,30,6,18,27,24,16,20,5,2,4,5,14,19,23,20,16,18,4,6,13,5,24,18,10,30,16,16,11,16,18,17,3,29,21,26,10,15,13,8,4,9,19,18,3,16,30,11,22,14,14,22,11,3,4,5,3,2,3,26,5,5,21,18,25,21,19,26,22,11,16,15},
    }

    local model = zdnn.FrozenLookupTable(db)

    for _, indices in ipairs(indices_list) do
        local out = model:forward( torch.LongTensor(indices) )

        tester:assertTableEq( 
            out:size():totable(), 
            { #indices, dim },
            1e-7,
            "output tensor size mismatch with input"
        )
        for i = 1,#indices do
            tester:assertTensorEq(
                out[i],
                db[ indices[i] ],
                1e-7,
                "output tensor error"
            )
        end
    end
    -- add test code here, using tester:asserteq methods
end


function test.FrozenLookupTableMaskedZero()
    local dim = 40
    local db = torch.rand( 30, dim )
    
    local indices_list = {
        {1,21,30,24,16,0,1,9,10}, 
        {25,12,7,24,0,8,6,19,15,1},
        {0,0,6,2,11,30,29,3,14,8,8,23,25,0,0,19,25,14,3,4,17,14,20,23,6,18,16,24,10,10,19,30,6,18,27,24,16,20,5,2,4,5,14,19,23,20,16,18,4,6,13,5,24,18,10,30,16,16,11,16,18,17,3,29,21,26,10,15,13,8,4,9,19,18,3,16,30,11,22,14,0,22,11,3,4,5,3,2,3,26,0,0,21,18,25,21,19,26,22,11,16,15,0,0},
    }

    local model = zdnn.FrozenLookupTable(db,true)

    local zero = torch.zeros(dim)

    for _, indices in ipairs(indices_list) do
        local out = model:forward( torch.LongTensor(indices) )

        tester:assertTableEq( 
            out:size():totable(), 
            { #indices, dim },
            1e-7,
            "output tensor size mismatch with input"
        )
        for i = 1,#indices do
            if indices[i] == 0 then
                tester:assertTensorEq(
                    out[i],
                    zero,
                    1e-7,
                    "output tensor shoule be zero"
                )
            else
                tester:assertTensorEq(
                    out[i],
                    db[ indices[i] ],
                    1e-7,
                    "output tensor error"
                )
            end
        end
    end
    -- add test code here, using tester:asserteq methods
end




-- add more tests by adding to the 'test' table


return tester:add(test):run()
