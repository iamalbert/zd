local tds = require 'tds'
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

string.trim = function(s)
    return (s:gsub("^%s*(.-)%s*$", "%1"))
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

function table.takeone(tbl)
    for k,v in pairs(tbl) do
        return k, v
    end
    return nil, nil
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
    isArray    = function (t)
      local i = 0
      for _ in pairs(t) do
          i = i + 1
          if t[i] == nil then return false end
      end
      return i
    end,
    isArrayOfTensors = function (t)
        if torch.isTensor(t) and t:dim() > 0 then 
          return t:size(1)
        else
          local i = 0
          for _ in pairs(t) do
            i = i + 1
            if not torch.isTensor(t[i]) then 
              return false 
            end
          end
          return i
        end
        return false
    end,

    isArrayLike = function(obj)
        if torch.isTensor(obj) then
            if obj:nElement() > 0 then
                return obj:size(1)
            else
                return false
            end
        elseif torch.type(obj) == "tds.Vec" then
            if #obj > 0 then
                return #obj
            else
                return false
            end
        else
            local s = zd.util.isArray(obj)
            if s ~= false and s > 0 then
                return s
            else
                return false
            end
        end
    end,

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

    recursive_find_tensor = function(obj, func)
        if zd.util.isTensor(obj) then
            return func(obj)
        elseif zd.util.isTable(obj) then
            for k,v in pairs(obj) do
                obj[k] = zd.util.recursive_find_tensor(v, func)
            end
            return obj
        else
            return obj
        end
    end,

    recursive_call_tensor = function(obj, func)
        if zd.util.isTensor(obj) then
            func(obj)
        else
            for k,v in pairs(obj) do
                zd.util.recursive_call_tensor(v, func)
            end
        end
    end,

    template_until_tensor = function(obj, new, f)
        if torch.isTensor(obj) then  
            new = new or obj.new()
            f(obj, new)
            return new
        else -- is an table
            new = new or {}
            for k,v in pairs(obj) do
                new[k] = zd.util.template_until_tensor(v, new[k], f)
            end
            return new
        end
    end,

    template_take_array = function(obj, idx, new )
        assert(idx ~= nil, "idx shall not be nil")
        return zd.util.template_until_array(obj, new, function(o,n)
            n[idx] = o[idx]
        end)
    end,

    cloneType = function(obj)
        if zd.util.isTensor(obj) then
            return obj.new()
        elseif torch.type(obj) == 'tds.Vec' then
            return tds.Vec()
        elseif torch.type(obj) == 'tds.Hash' then
            return tds.Hash()
        else
            return {}
        end
    end,



    template_until_array = function(obj, new, f)
        local sz = zd.util.isArrayLike(obj)
        new = new or zd.util.cloneType(obj)
        if sz == false then  -- is an hash-table
            for k,v in pairs(obj) do
                new[k] = new[k] or {}
                zd.util.template_until_array(v, new[k], f)
            end
        else -- is an array
            f(obj, new)
        end
        -- print(obj, new)
        return new
    end,

    recursive_cuda = function(obj)
        return zd.util.recursive_find_tensor(obj, function(o)
            return o:cuda()
        end)
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

