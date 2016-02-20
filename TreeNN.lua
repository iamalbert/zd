require 'nn'

local Tree, parent = torch.class('nn.Tree', 'nn.Container')

function Tree:__init( module )
    parent.__init(self)
    self:add( module )
    self.module = self.modules[1]
    self._clones = {}
end

function Tree:updateOutput( input )
    input:dfs( nil, function(t)
        if t:isTerminal() then
            t:set('output',  t:get('input') )
        else
            if t:n_child() == 1 then
                t:set('input', t:child(1):get('output'):clone() )
                t:set('output', t:get('input'):clone() )
            elseif t:n_child() == 2 then
                t:set('input', {
                    t:child(1):get('output'):clone(),
                    t:child(2):get('output'):clone()
                })
                t._module = self.module:clone('weight','bias')
                table.insert( self._clones, t._module )
                local out = t._module:forward( t:get('input')  )
                t:set('output',  out )
            else
                error "tree must be binary"
            end
        end
        -- print(t._data) 
    end)
    return input:get('output')
end

function Tree:_backpropagate( input, gradOutput, func )
    input:dfs( function( t, lv, ic ) 
        local go
        if t == input then -- is root
            go = gradOutput
        else
            if t:parent():n_child() == 2 then
                go = t:parent():get('gradInput')[ic]
            else
                go = t:parent():get('gradInput')
            end
        end

         t:set('gradOutput', go )

        local Input
        local gi


        if t._module then
            Input = t:get('input')
            gi = func( t._module, Input , go ) 
        else
            gi = go
        end
        if gi then t:set('gradInput', gi) end
       --print( t._data )
    end)
    return input
end

function Tree:backward(input, gradOutput) 
    self:_backpropagate( input, gradOutput, function( module, In, go )
        local gi = module:updateGradInput( In, go )
        module:accGradParameters( In, go )
        return gi
    end)
    return input
end

function Tree:updateGradInput(input, gradOutput)
    self:_backpropagate( input, gradOutput, function( module, In, go )
        return module:updateGradInput( In, go )
    end)
    return input
end

function Tree:accGradParameters(input, gradOutput, scale)
    self:_backpropagate( input, gradOutput, function( module, In, go )
        -- print( {In=In, go=go})
        module:accGradParameters( In, go, scale )
    end)
    return input
end

function Tree:updateParameters(lr)
    for _, module in ipairs(self._clones) do
        module:updateParameters(lr)
    end
end

Tree.__tostring = nn.Sequential().__tostring

local TreeLinear, parent = torch.class('nn.TreeLinear', 'nn.Sequential')
function TreeLinear:__init( inputDim )
    parent.__init(self)
    self._inputDim = inputDim
    self._outputDim = inputDim

    self
    :add( nn.ParallelTable()
        :add( nn.Linear(inputDim, inputDim, false ) )
        :add( nn.Linear(inputDim, inputDim, false) )
    )
    :add( nn.CAddTable() )
end

