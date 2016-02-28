local Class, Parent = torch.class('zdnn.JoinTable', 'nn.JoinTable')

function Class:__init(dim,nInputDims) 
	Parent.__init(self,dim,nInputDims)
end
function Class:updateOutput(input)
   self.output:typeAs(input[1]):resize(self.size)
   return Parent.updateOutput(input)
end

function Class:updateGradInput(input, gradOutput)
	if gradOutput == nil or gradOutput:dim() == 0 then
		self.gradInput = nil
		return self.gradInput
	else
		return Parent.updateGradInput(input, gradOutput)
	end
end
