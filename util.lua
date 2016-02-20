local escapepatter = {
    ['\n'] = '\\n',
    ['\t'] = '\\t',
    ['\r'] = '\\r',
    ['\a'] = '\\a',
    ['\\'] = '\\\\',
}
local quotepattern = '(['..("%^$().[]*+-?"):gsub("(.)", "%%%1")..'])'
string.ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-.$"

string.at = function( str, i )
    return string.sub(str,i,i)
end

string.escape = function(str)
    local s = ""
    for i = 1,#str do
        local c = str[i] 
        if escapepatter[c] ~= nil then
            s = s .. escapepatter[c]
        else
            s = s .. c
        end
    end
    return s
end

function string.startswith( self, prefix )
    return string.match( self, "^".. prefix ) ~= nil
end

function string.endswith( self, suffix )
    return string.match( self, suffix .. '$') ~= nil
end

function string.quote(str)
    return str:gsub(quotepattern, "%%%1")
end
function string.pos ( str, c )
    for i = 1,#str do
        if string.at(str,i) == c then
            return i
        end
    end
    return nil
end
function table.contains(tbl, val)
    for k,v in pairs(tbl) do
        if v == val then
            return k
        end
    end
    return nil
end

function table.count( tbl ) 
    local cnt = 0
    for i, v in pairs(tbl) do 
      cnt = cnt + 1
    end
    return cnt
end

function table.index( tbl, indices )
    local ret = {}
    for i=1, zd.util.get_size(indices) do
        table.insert(ret, tbl[ indices[i] ] )
    end
    return ret
end

function table.deepcopy(orig)
    local copy
    if zd.util.isTable(orig) then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[table.deepcopy(orig_key)] = table.deepcopy(orig_value)
        end
        setmetatable(copy, getmetatable(orig))
    elseif zd.util.isTensor(orig) then
      copy = orig:clone() 
    else -- number, string, boolean, etc
      copy = orig
    end
    return copy
end

function table.shift(tbl)
    local v = table.remove(tbl,1)
    return v, tbl
end
  
function table.map( func, tbl )
    local newtbl = {}
    for i,v in pairs(tbl) do
        newtbl[i] = func(v)
    end
    return newtbl
end

function table.transpose ( tbl )
    local ret = {}
    for i,v in pairs(tbl) do
      ret[v] = i
    end
    return ret
end

local util = torch.class('zd.util')

function util:__init()
    error "should not be initiated"
end

local util_impl = {
    isTable    = function(obj) return type(obj) == "table"    end,
    isNumber   = function(obj) return type(obj) == "number"   end,
    isString   = function(obj) return type(obj) == "string"   end,
    isFunction = function(obj) return type(obj) == "function" end,
    isTensor   = torch.isTensor,

    make_immutable = function (tbl)  
      return setmetatable({}, {  
        __index = tbl,  
        __newindex = function(t, key, value)  
          error("attempting to change constant " ..  
            tostring(key) .. " to " .. tostring(value), 2)  
        end  
      })  
    end,

    get_size = function(obj)
        if zd.util.isTensor(obj) then
            assert( obj:dim() > 0, "get a zero-dimension tensor" )
            return obj:size(1)
        else
            return #obj
        end
    end,

    nop = function (...) end,
    identity = function(...) return ... end,

    callunlessnil = function( func , ... )
        if func ~= nil then
            return func(...)
        else
            return nil
        end
    end
}

for name, func in pairs(util_impl) do
    util[name] = func
end
