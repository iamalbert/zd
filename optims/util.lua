zd.optim = {}

function zd.optim._perform_l2reg(x, dfdx, l2reg)
    if l2reg ~= nil and l2reg ~= 0 then
        dfdx:add( l2reg, dfdx )
    end
end

function zd.optim._perform_momentum(x, dfdx, config, state)
	-- copy from torch/optim.sgd

	local mom  = config.momentum or 0 
	local damp = config.dampening or mom
	local nesterov = config.nesterov or false

	if mom ~= 0 then
	    if not state.dfdx then
		 	state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
	    else
	  		state.dfdx:mul(mom):add(1-damp, dfdx)
	    end
		if nesterov then
			dfdx:add(mom, state.dfdx)
	    else
			dfdx:copy(state.dfdx)
	    end
	end
end


