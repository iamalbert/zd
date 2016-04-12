#!/usr/bin/env th

require 'zd'


local test = torch.TestSuite()

local tester = torch.Tester()


function test.RecursiveCuda()
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

function test.isArrayLike()
    local f = zd.util.isArrayLike
    local tds = require 'tds'

    tester:assert( f {} == false, "{} should be false")
    tester:assert( f {1,2,3,40} == 4, "{1,2,3,4} should be 4")
    tester:assert( f(tds.Vec{1,2,3,40}) == 4, "tds.Vec{1,2,3,4} should be 40")

    tester:assert( f{{1,2,3,40}} == 1, "{{1,2,3,4}} should be 1")
    tester:assert( f{{1,2,3,40}, {5,6,7,8} } == 2, 
        "{{1,2,3,4},{5,6,7,8}} should be 2")

    tester:assert( f(torch.rand(3)) == 3, "torch.rand(3) should be 3")
    tester:assert( f(torch.rand(2,3,5,6)) == 2, "torch.rand(2,3,5,6) should be 2")
end

function test.TemplateArrayLike()
    local tds = require 'tds'
    local rand = torch.rand(3,4,5)
    local template = {
        input = rand,
        deep = {
            a = {
                b = { 0, -1, -2},
                c = {
                    e = { -3, -4, -5 }
                },
            },
            d = tds.Vec { 20, 30, 40 }
        },
        target = torch.LongTensor{ -1,-2,-3 }
    }

    local target = {
        input = rand[1],
        deep = {
            a = {
                b = { 0 },
                c = {
                    e = { -3 }
                },
            },
            d = tds.Vec { 20 }
        },
        target = torch.LongTensor{ -1 }
    }

    local pred = zd.util.template_take_array(template, 1)

    tester:assert(pred ~= nil, "pred shall not be nil")
    tester:assert(pred.input ~= nil, "pred.input shall not be nil")
    tester:assert(pred.deep ~= nil,  "pred.deep shall not be nil")
    tester:assert(pred.deep.a ~= nil,  "pred.deep shall not be nil")
    tester:assert(pred.deep.a.b ~= nil,  "pred.deep shall not be nil")
    tester:assert(pred.deep.a.c ~= nil,  "pred.deep shall not be nil")
    tester:assert(pred.deep.a.c.e ~= nil,  "pred.deep shall not be nil")
    tester:assert(pred.deep.d ~= nil,  "pred.deep shall not be nil")
    tester:assert(pred.target ~= nil,  "pred.deep shall not be nil")

    tester:assertGeneralEq(pred.input[1], template.input[1], "pred.input shall not be nil")
    tester:assertGeneralEq(pred.deep.a.b[1], template.deep.a.b[1],  "pred.deep shall not be nil")
    tester:assertGeneralEq(pred.deep.a.c.e[1] , template.deep.a.c.e[1],  "pred.deep shall not be nil")
    tester:assertGeneralEq(pred.deep.d[1] , template.deep.d[1],  "pred.deep shall not be nil")
    tester:assertGeneralEq(pred.target[1] , template.target[1],  "pred.deep shall not be nil")


end

-- add more tests by adding to the 'test' table

return tester:add(test):run()
