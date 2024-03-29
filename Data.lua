do
    local Class = torch.class('zd.Sampler')
    --[[

    --]]

    function Class:__init(config)
        assert( config,  "require a config table"  )
        config.template = config.template or config.source
        assert( config and zd.util.isTable(config.template), 
            "field `template' (or `template') is required"  )
        self._template = config.template
        self._config = config
        self._cuda = config.cuda
        self._callback = config.callback
    end

    function Class:cuda(opt)
        self._cuda = not not opt
        return self
    end

    function Class:reset()
        self.state = {
            index = 1,
        }
        local max = self:rewind()
        assert( zd.util.isNumber(max), "::rewind() must return a number,"
            .. "got `" .. type(max) .. "' instead.")
        max = math.floor(max)
        self.state.max = max
        return max
    end

    function Class:batchSize()
        return math.floor( tonumber( self._config.batch or 0 ) )
    end

    function Class:next()
        local ret = self:current(self.state.index)

        assert( ret ~= nil, "::current() return value is nil" )
        if self._callback then
            ret = self._callback(ret)
            assert( ret ~= nil, "callback() return value is nil" )
        end

        local state = self.state
        local oldindex = state.index
        state.index = state.index + 1

        if self._cuda then
            ret = zd.util.recursive_cuda(ret)
        end

        return ret,oldindex
    end

    function Class:rewind()
        error "not implemented"
    end
    function Class:current()
        error "not implemented"
    end
    function Class:finished(state)
        local S = self.state
        if S and (S.index > S.max) then
            return true
        else
            return false
        end
    end

    function Class:each( func )
        self:reset()
        repeat 
            func( self:next() )
        until self:finished()
    end
end

do 
    local Class, Parent = torch.class('zd.SmartIterator', 'zd.Sampler')
    function Class:__init(config)
        Parent.__init(self, config)
    end
    function Class:rewind()
        local state, template, config = self.state, self._template, self._config

        local sizes = {}
        zd.util.recursive_call_tensor( template, function(t)
            assert( t:dim() > 0, "find a zero-dim tensor in template")
            table.insert( sizes, t:size(1) )
        end)

        for i = 2, #sizes do
            local sz = sizes[i]
            assert( sz == sizes[1], 
                "tensor size inconsistent: " .. table.concat(sizes, ','))
        end

        local max = sizes[1]
        self.length = max

        if config.shuffle then
            self.order = torch.randperm(1, max):long()
        else
            self.order = torch.range(1, max):long()
        end

        if self:batchSize() > 0 then
            local bs = self:batchSize()
            max = math.ceil(max / bs) 
        end

        return max
    end
    function Class:getBatch(i, yield)
        local bs = self:batchSize()

        local first = (i-1)*bs + 1 
        local last  = i * bs

        if last > self.length then
            last = self.length
        end

        local idx = self.order:sub( first, last )

        return zd.util.template_until_tensor( self._template, yield, 
            function(template,new)
                new:set(template:index(1,idx))
            end
        )
    end
    function Class:getNoBatch(i, yield)
        local idx = self.order[i]
        return zd.util.template_until_tensor( self._template, yield, 
            function(template,new)
                new:set(template[i])
            end
        )
    end
    function Class:current(i)
        self.yield = self.yield or {}
        if self:batchSize() > 0 then
            self.yield = self:getBatch(i, self.yield)
        else
            self.yield = self:getNoBatch(i, self.yield)
        end
        return self.yield
    end
end

do 
    local Class, parent = torch.class('zd.Iterator', 'zd.Sampler')

    function Class:rewind()
        local template = self._template
        local config = self._config
        
        local state = self.state
        state.max = nil 

        for name, db in pairs(template) do
            if state.max == nil then
                state.max = zd.util.get_size( db )
            else
                if zd.util.get_size(db) ~= state.max then
                    error("inconsitent length of data templates")
                end
            end
        end
        assert( state.max, "no data given")

        local order

        if config.shuffle then
            order = torch.randperm( state.max ):long()
        else
            order = torch.range(1, state.max ):long()
        end

        if self:batchSize() > 0 then
            local bs = self:batchSize()
            local pad_length = math.ceil( state.max / bs ) * bs - state.max

            if pad_length > 0 then
                order:resize( state.max + pad_length )
                order:sub( state.max+1, state.max+pad_length):fill(0)
            end

            state._pad_length = pad_length

            state.max = (state.max + pad_length ) / bs
            order:resize( state.max, bs )
        else
            order:resize( state.max, 1 )
        end

        state.order = order
        return state.max
    end

    function Class:current()
        local state, template =  self.state, self._template
        local cur = state.order[ state.index ]

        if state.index == state.max and state._pad_length ~= nil then
            cur = cur:sub(1, - state._pad_length - 1)
        end

        local ret = {}
        for name, data in pairs(template) do
            local datum
            if zd.util.isTensor(data) then
                if self:batchSize() > 0 then
                    datum = data:index( 1, cur )
                else
                    datum = data:select( 1, cur[1] )
                end
            else -- data is table
                if self:batchSize() > 0 then
                    datum = table.index(data, cur)
                else
                    datum = data[ cur[1] ]
                end
            end
            ret[name] = datum
        end
        return ret
    end
end
