#!/usr/bin/env th


require 'zd'

local test = torch.TestSuite()

local tester = torch.Tester()

local inputs  = torch.rand( 883, 40, 500 )
local targets = torch.rand( 883, 1)

function test.no_shuffle()
    local iter = zd.Iterator {
        source = {
            input = inputs,
            target = targets,
        }
    }
    iter:reset()
    repeat
        local datum, i = iter:next()
        tester:assertTensorEq(datum.input, inputs[i], 1e-7,
            "entry.input not equal inputs[i]")
    until  iter:finished()
end

function test.no_shuffle_batch()
    local bs = 17
    local iter = zd.Iterator {
        source = {
            input = inputs,
            target = targets
        },
        batch = bs
    }
    iter:reset()
    repeat
        local datum, i = iter:next()
        start = (i-1) * bs + 1

        local len = bs
        if start + len - 1 > inputs:size(1) then
            len = inputs:size(1) - start + 1
        end
        tester:assertTensorEq(
            datum.input, 
            inputs:narrow(1, start, len),
            1e-7)
    until iter:finished()
end


function test.smart_iterator()
    local N, T1, T2, D = 100, 30, 10, 20
    local C = 11

    local seq1 = torch.rand(N,T1,D)
    local seq2 = torch.rand(N,T2,D)

    local target = torch.LongTensor(N, C)

    local iter = zd.SmartIterator {
        template = {
            input = { seq1, seq2 },
            target = target
        },
    }

    iter:reset()
    for i = 1, N do
        local yield = iter:next()

        local sample = {
            input = { seq1[i], seq2[i] },
            target = target[i]
        }
        tester:assertGeneralEq( sample.target, yield.target )
        tester:assertGeneralEq( sample.input[1], yield.input[1] )
        tester:assertGeneralEq( sample.input[2], yield.input[2] )
    end
end

function test.smart_iterator_batch()
    local N, T1, T2, D = 100, 30, 10, 20
    local C = 11

    local seq1 = torch.rand(N,T1,D)
    local seq2 = torch.rand(N,T2,D)

    local target = torch.LongTensor(N, C)

    local batch = 30
    local iter = zd.SmartIterator {
        template = {
            input = { zd.Source(seq1), zd.Source(seq2) },
            target = zd.Source(target)
        },
        batch = batch
    }

    iter:reset()
    for i = 1, math.floor(N/batch)*batch, batch do
        local idx3 = {{i,i+batch-1}, {}, {}}
        local idx2 = {{i,i+batch-1}, {}}
        local yield = iter:next()

        local sample = {
            input = { seq1[idx3], seq2[idx3] },
            target = target[idx2]
        }
        tester:assertGeneralEq( sample.target, yield.target )
        tester:assertGeneralEq( sample.input[1], yield.input[1] )
        tester:assertGeneralEq( sample.input[2], yield.input[2] )
    end
    do 
        local yield = iter:next()
        local idx3 = {{91,100}, {}, {}}
        local idx2 = {{91,100}, {}}
        local sample = {
            input = { seq1[idx3], seq2[idx3] },
            target = target[idx2]
        }
        tester:assertGeneralEq( sample.target, yield.target )
        tester:assertGeneralEq( sample.input[1], yield.input[1] )
        tester:assertGeneralEq( sample.input[2], yield.input[2] )
    end
end
-- add more tests by adding to the 'test' table

return tester:add(test):run()
