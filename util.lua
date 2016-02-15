local quotepattern = {
    ['\n'] = '\\n',
    ['\t'] = '\\t',
    ['\r'] = '\\r',
    ['\a'] = '\\a',
    ['\\'] = '\\\\',
}

string.at = function( str, i )
    return string.sub(str,i,i)
end

string.ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-.$"

string.quote = function(str)
    local s = ""
    for i = 1,#str do
        local c = str[i] 
        if quotepattern[c] ~= nil then
            s = s .. quotepattern[c]
        else
            s = s .. c
        end
    end
    return s
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


