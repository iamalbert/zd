return function(zd)
  local Expr = zd:class('Expr')
  local util = zd.util

  Expr.events = {
    "before_start"      ,
    "before_epoch"      , 
    "before_example"    ,
    "before_backward"   ,
    "before_feedback"   ,
    "before_update"     ,
    "after_epoch"       ,
    "after_example"     ,
    "after_update",
  }

  function Expr:__init(arg)
    assert( arg and type(arg)=="table", "must give me arguments")

    local required = {
      "model", 
      "criterion", 
      "data", 
    }

    local optional = {
      optimizer        = optim.sgd,

      state            = {},
      feedback         = false,
      grad_param       = function() return util.identity end,
      cuda             = false,
      file             = false,
      max_epoch        = -1,
      shuffle          = false,
      batch_size       = 1
    }

    for i,v in ipairs(Expr.events) do
      optional[v] = util.nop
    end

    for i,v in ipairs(required) do
      assert( arg[v], "lack of `" .. v .. "' in Expr constructor")
      self[v] = arg[v]
    end

    for i,v in pairs(optional) do
      self[i] = arg[i] or v
      -- print( i, self[i] )
    end

    if self.cuda then
      self.model = self.model:cuda()
      self.criterion = self.criterion:cuda()
    end
  end

  function Expr:forward( example )
    example.output = self.model:forward(example.input)
    return example.output
  end


  function Expr:backward(example)
    example.gradOutput = self.criterion:backward( example.output, example.target )
    example.gradInput = self.model:backward( example.input, example.gradOutput )
  end

  function Expr:update(state, params, gradParams)
    -- print("state", state)
    self.optimizer( 
      function(p)
        if params ~= p then params:copy(p) end

        local l2reg = state.optim_config.l2reg 
        if l2reg ~= nil and l2reg ~= 0 then
            gradParams:add( l2reg, params )
        end
        return state.loss, gradParams
      end,
      params, 
      state.optim_config, state.optim_state
    )

    self.model:zeroGradParameters()
  end

  local function import_event( events, ... )
    -- print(events)
    local args = {...}
    local ret  = {}


    for _, e in ipairs(events) do
      for _,model in ipairs(args) do
        local cb = model[e]
        if cb ~= nil then
          ret[e] = cb
          break
        end
      end
      if ret[e] == nil then
        ret[e] = zd.util.nop
      end
    end

    return ret
  end

  function Expr:_do_epoch( expr_state, evt )
    local state = expr_state

    local params, gradParams 

    if state.mode == 'train' then
      params, gradParams = self.model:getParameters()

      state.optim_state = table.deepcopy( state.init_optim_state or {} )
      self.model:zeroGradParameters()

      self.model:training()
    else
      self.model:evaluate()
    end

    state.data_iter_obj, state.max_example 
                      = state.sampler(self.data[state.mode])

    for _, example in zd.data_iter(state.data_iter_obj) do
      state.n_example = state.n_example + 1

      if false ~= evt.before_example(self, example, state) then
        self:_do_example(example, state, evt, params, gradParams)
      end

      evt.after_example(self, example, state)
    end

  end

  function Expr:_do_example( example, state, evt, params, gradParams )
    self:forward( example )

    if false ~= evt.before_feedback(self, example, state) then
      self:add_feedback(example, state)
    end

    state.loss = self.criterion:forward(example.output, example.target)
    
    if state.mode == 'train' then
      if evt.before_backward(self,example, state) ~= false then
        self:backward( example )
        if evt.before_update(self, example, state) ~= false then
          self:update(state, params, gradParams)
          evt.after_update(self, example, state )
        end
      end
    end
  end

  function Expr:start( expr_state )

    local state = expr_state or {}
    assert( state.sampler, "must provide sampler" )

    local evt = import_event( Expr.events, expr_state, self )

    state.n_epoch          = 0
    state.mode             = state.mode or 'train'

 
    local max_epoch 

    if state.max_epoch ~= nil then
      max_epoch = tonumber(state.max_epoch)
    elseif self.max_epoch ~= nil then
      max_epoch = tonumber(self.max_epoch)
    else
      max_epoch = -1
    end

    if state.mode == 'train' then
      self.model:training()
      state.init_optim_state = self.init_optim_state or {}
      state.optim_config     = self.optim_config or {}
    else
      self.model:evaluate()
    end

    evt.before_start(self, state)

    -- print("max_epoch", max_epoch, state )

    while max_epoch < 0 or state.n_epoch < max_epoch do

      state.n_example = 0
      state.n_epoch = state.n_epoch + 1


      if self.feedback then self.feedback:zero() end

      if false == evt.before_epoch(self, state) then
        break
      end

      self:_do_epoch( state, evt )

      if false == evt.after_epoch(self, state) then
        break
      end
    end

    self:save_model(state)
  end

  function Expr:add_feedback( example, state )
    local sampler = state.sampler
    if sampler and self.feedback then
      if sampler.is_batch then 
        self.feedback:batchAdd( example.output, example.target ) 
      else
        self.feedback:add( example.output, example.target )
      end
    end
    return self
  end

  function Expr:save_model(state)
    local filename = nil
    if util.isFunction( self.file ) then
      filename = self.file(state.n_epoch)
    elseif util.isString( self.file) then
      filename = self.file 
    end
    if filename then
      print( "save model to", filename )
      torch.save( filename, self.model )
    end
  end

end
