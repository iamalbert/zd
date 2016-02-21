#!/usr/bin/env th

require 'totem'

local test = {}

local tester = totem.Tester()


function test.RecursiveFindTensor()
    require 'cutorch'

    local i = {
        torch.rand(30, 50),
        {
            f = torch.rand(40),
            s = {
                torch.rand(2,3,4,5),
                torch.rand(190)
            }
        }
    }

    local o = zd.util.recursive_find_tensor(i, function(t)
        return t:cuda()
    end)

    tester:asserteq( type(o), type(i) )

    tester:assertTensorEq( i[1], o[1]:typeAs(i[1]), 1e-7 )
    tester:assertTensorEq( i[2].f, o[2].f:typeAs(i[2].f), 1e-7 )
    tester:asserteq( type(i[2].s), type(o[2].s) )

    tester:assertTensorEq( i[2].s[1], o[2].s[1]:typeAs(i[2].s[1]), 1e-7 )
    tester:assertTensorEq( i[2].s[2], o[2].s[2]:typeAs(i[2].s[2]), 1e-7 )

end


-- add more tests by adding to the 'test' table


return tester:add(test):run()
