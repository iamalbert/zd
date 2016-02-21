#!/usr/bin/env th

require 'nn'
require 'optim'
require 'totem'

require 'zd'

local test = {}

local tester = totem.Tester()

local n_data, inDim, outDim = 100, 30, 40
local data = torch.rand( n_data, inDim )
local ans  = torch.rand( n_data, outDim )

local model = nn.Linear(inDim, outDim)
local criterion = nn.AbsCriterion()
local feedback = optim.ConfusionMatrix(outDim)

function test.Propagate()
    local called = {}
    for _, event in ipairs(zd.Evaluator._allow_event_list) do
        called[event] = 0
    end

    local iter = zd.Iterator {
        source = {
            input = data,
            target = ans
        }
    }
    local target = 1

    local incre = function(name)
        return function()
            called[name] = called[name] + 1
        end
    end

    local evaluator = zd.Evaluator {
        model = model,
        events = {
            before_epoch = incre 'before_epoch',
            before_feedback = incre 'before_feedback',
            before_loss = incre 'before_loss',
            
            after_epoch = incre 'after_epoch',

            after_example = function(self, state, example )

                tester:assertTensorEq(example.input, data[state.n_example],
                    1e-7, "wrong input for n_example=" .. state.n_example
                )
                tester:assertTensorEq(example.target, ans[state.n_example],
                    1e-7, "wrong target for n_example=" .. state.n_example
                )

                tester:asserteq( example.output:dim(), 1, 
                    "output dimension wrong"
                )
                tester:asserteq( example.output:size(1), outDim, 
                    "output size wrong" 
                )
                
                called.after_example = called.after_example + 1
            end,
            before_example = function(self, state, example)
                tester:assertTensorEq(example.input, data[state.n_example],
                    1e-7, "wrong input for n_example=" .. state.n_example
                )
                tester:assertTensorEq(example.target, ans[state.n_example],
                    1e-7, "wrong target for n_example=" .. state.n_example
                )
                called.before_example = called.before_example + 1
            end
        }
    }


    local state = evaluator:run(iter)
    tester:asserteq( called.before_epoch, 1 ,
        "`before_epoch' should be called once")
    tester:asserteq( called.after_epoch, 1 ,
        "`after_epoch' should be called once")

    tester:asserteq( called.before_feedback, n_data, 
        "`before_example' does not make every example")
    tester:asserteq( called.before_loss, n_data, 
        "`before_loss' does not make every example")
    tester:asserteq( called.before_feedback, n_data, 
        "`before_feedback' does not make every example")
    tester:asserteq( called.after_example, n_data, 
        "`after_example' does not make every example")

end


-- add more tests by adding to the 'test' table


return tester:add(test):run()
