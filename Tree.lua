local Tree = torch.class('zd.Tree')

function Tree:__init()
    self._parent = nil
    self._child = {}
    self._data = {}
end

function Tree:child(n) return self._child[n] end
function Tree:parent() return self._parent end

function Tree:get(prop)
    return self._data[prop]
end
function Tree:set(prop, value)
    self._data[prop] = value
    return self
end

function Tree:add( tree )
    table.insert(self._child, tree)
    tree._parent = self
end

function Tree:isRoot() 
    return self._parent == nil
end
function Tree:isTerminal() 
    return self:n_child() == 0 
end
function Tree:isPreterminal() 
    return self:n_child() == 1 and self:child(1):isTerminal()
end

function Tree:n_child()
    return #self._child
end

function Tree:dfs(func1, func2, lv, idx_child, parent)
    lv = lv or 1
    idx_child = idx_child or 1
    if func1 then func1(self, lv, idx_child, parent) end

    for k,tree in ipairs(self._child) do
        tree:dfs(func1, func2, lv+1, k, self )
    end

    if func2 then func2(self, lv, idx_child, parent) end
end

function Tree:getTerminals()
    return self:yield( function(t) return t end)
end

function Tree:getAllNodes()
	local ret = {}
	self:dfs( function(t)
		table.insert(ret, t )
	end)
	return ret
end

function Tree:getTerminal(n)
    return self:yield( function(t) return t end) [n]
end

function Tree:yield( func )
    local ret = {}
    self:dfs(function(self)
        if self:isTerminal() then
            if type(func) == "function" then
                table.insert( ret, func(self) )
			elseif type(func) == "string" then
				table.insert( ret, self._data[func] )
            else
                table.insert( ret, self._data )
            end
        end
    end)
    return ret
end

function Tree:yieldPreterminals( func )
    local ret = {}
    self:dfs(function(self)
        if self:isPreterminal() then
            if func then
                table.insert( ret, func(self) )
            else
                table.insert( ret, self._data )
            end
        end
    end)
    return ret
end

function Tree:level()
    local idx = 1
    local odx = 1
    self:dfs( function(t) 
        if t == self then
            t._level = t._level or 1
        else
            t._level = t:parent()._level + 1
        end
        t._index = idx
        idx = idx + 1
    end, function(t)
        t._outindex = odx
        odx = odx + 1
    end)
end

function Tree:n_node()
    self:dfs( nil, function(t)
        t._cnt = 1
        for _, c in ipairs(t._child) do
            t._cnt = t._cnt + (c._cnt or 0)
        end
    end)
    return self._cnt
end

function Tree:lowestCommonAncestor(a,b)
    if self._level == nil or a._level == nil or b._level == nil then
        self:level()
    end
    while a ~= b do
        if a._level > b._level then
            a = a:parent()
        else
            b = b:parent()
        end
    end
    return a
end

function Tree:__tostring()
    local ret = ''
    -- self:level()
    self:dfs( function(self, lv)
        local s = xlua.table2string(self._data)
        ret = ret .. string.rep('    ', lv - 1) .. s .. '\n'
    end)
    return ret
end

function Tree:toBinary()
    self:dfs( function(t, lv)
        if t:n_child() > 2 then
            local left_child = t:child(1)
            table.remove( t._child, 1 )

            local right_child = t.new()
            right_child._child = t._child
            right_child:set('lab',t:get('lab') .. '@' )
            for _,c in ipairs(right_child._child) do
                c._parent = right_child
            end

            t._child = {}
            t:add( left_child )
            t:add( right_child )
        end
    end)
    return self
end


local alpha = string.ALPHA .. "_~!@#$%^&*_+1234567890?><,./\\|"
local OTHERWISE = zd.FSM.OTHERWISE

local trans = {
    INIT = {
        ["("] = function( conf, stack ) 
            table.insert( stack, {"LP", "("} )
            return "LABEL"
        end,
        [")"] = function(conf, stack)
            table.insert(stack, {"RP", ")"} )
            return "INIT"
        end,
        [OTHERWISE] = function( conf, stack )
            conf.lab = conf:current()
            return "LABEL"
        end,
        ["\n\r\t "] = function( conf, stack )
            return "INIT"
        end
    },
    LABEL = {
        [" \n\r\t"] = function( conf, stack )
            table.insert( stack, {"LAB", conf.lab} )
            conf.lab = nil
            return "INIT"
        end,
        [OTHERWISE] = function( conf, stack )
            conf.lab = conf.lab or ''
            conf.lab = conf.lab .. conf:current()
            return "LABEL"
        end,
        [")"] = function(conf, stack)
            table.insert( stack, {"LAB", conf.lab} )
            conf.lab = nil
            return "INIT", false
        end,

    }
}
local tt = {
   LABEL = {
        [{"LAB"}] = function(S) 
            local t = S.root
            local s = S:current()[2]
            if t:get('lab') == nil then
                t:set('lab', s)
            else
                t:set('value', s) 
            end
            return "LABEL"
        end,
        [{"LP"}] = function(S)
            local newt = zd.Tree()
            if S.root ~= nil then
                S.root:add( newt )
                S.root = newt
            else
                S.root = newt
            end
            return "LABEL"
        end,
        [{"RP"}] = function(S)
            if not S.root:isRoot() then
                S.root = S.root:parent()
            end
            return "LABEL"
        end
    }

}

function Tree.parsePenn(str, debug)
	local lexer = zd.FSM.run( trans, zd.FSMConfig(str, "INIT", {
		debug=debug
	} ) )
	local parser = zd.FSM.run( tt, zd.FSMConfig(lexer.stack, "LABEL", {
		debug = debug,
		cur = function( self, t ) return t[1] end
	}))
	return parser.root
end

