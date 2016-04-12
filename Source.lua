do
    local Class, Parent = torch.class('zd.Source')

    function Class:__init(obj, pad_value)
        if torch.isTensor(obj) then
            self.type = 'tensor'
            assert( obj:dim() > 0, "empty tensor is in valid")
        elseif torch.type(obj) == 'tds.Vec' 
                or torch.type(obj) == 'tds.Hash' then
            self.type = 'tds'
        elseif type(obj) == "table" then
            self.type = 'table'
        else
            error "only support Tensor, tds.Vec, or table"
        end

        self.obj = obj
        self._size = zd.util.isArrayOfTensors(self.obj)
        assert( self._size ~= false, "must be array-like of tensors")
        assert( self._size > 0, "must have at least 1 tensor ")

        self.pad_value = pad_value or 0
    end

    function Class:size()
        return self._size
    end

    function Class:get(i)
        return self.obj[i]
    end

    function Class:getBatch(indices)
        local obj = self.obj
        if self.type == 'tensor' then
            return self.obj:index(1, indices)
        else
            local maxlen = 0
            for i = 1, indices:size(1) do
                local idx = indices[i]
                local len = obj[ idx ]:size(1)
                if len > maxlen then maxlen = len end
            end
            local size = obj[1]:size():totable()
            size[1] = maxlen
            table.insert(size, 1, indices:size(1) )
            size = torch.LongStorage(size)

            local ret = self:getBatchReturn(size)
            for i = 1, indices:size(1) do
                local idx = indices[i]
                local p = obj[ idx ]
                self:setEntryOfBatch(ret, i, p )
            end
            return ret
        end
    end
    function Class:getBatchReturn(size)
        return self.obj[1].new():resize(size):fill(self.pad_value)
    end
    function Class:setEntryOfBatch(ret, i, p)
        ret:sub( i,i,1, p:size(1) ):copy(p)
    end
end
do
    local Class, Parent = torch.class('zd.MaskSource', 'zd.Source')
    function Class:getBatchReturn(size)
        print("called", size)
        return torch.ByteTensor():zeros(size)
    end
    function Class:setEntryOfBatch(ret, i, p)
        ret:sub( i,i,1,p:size(1) ):fill(1)
    end
end
