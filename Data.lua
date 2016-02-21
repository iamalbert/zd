do
    local Class = torch.class('zd.Sampler')

    function Class:__init()
    end

    function Class:reset()
        error "not implemented"
    end

    function Class:next(state)
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
    local Class, parent = torch.class('zd.Iterator', 'zd.Sampler')

    function Class:__init(config)
        assert( config,  "require a config table"  )
        assert( config and config.source, "field `source' is required"  )
        self._data = config.source
        self._config = config
    end

    function Class:batchSize()
        return math.floor( tonumber( self._config.batch or 0 ) )
    end

    function Class:reset()
        local data = self._data
        local config = self._config

        local state = {
            index = 1,
            max   = zd.util.get_size( data )
        }

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

        self.state = state
        return self.state, state.max
    end

    function Class:next()
        local state, data = self.state, self._data
        local cur = state.order[ state.index ]

        if state.index == state.max and state._pad_length ~= nil then
            cur = cur:sub(1, - state._pad_length - 1)
        end

        local ret
        if zd.util.isTensor(data) then
            if self:batchSize() > 0 then
                ret = data:index( 1, cur )
            else
                ret = data:select( 1, cur[1] )
            end
        else -- data is table
            if self:batchSize() > 0 then
                ret = table.index(data, cur)
            else
                ret = data[ cur[1] ]
            end
        end
        local oldindex = state.index

        state.index = state.index + 1

        return ret, oldindex


end
