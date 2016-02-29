local Class, Parent = torch.class('zdnn.JoinTable', 'nn.JoinTable')

function Class:__init(dim,nInputDims) 
	Parent.__init(self,dim,nInputDims)
end
function Class:updateOutput(input)
   Parent.updateOutput(self,input)
   self.output = self.output:type(input[1]:type())
   -- print(self.output:type(), input[1]:type() )
   return self.output
end

function Class:updateGradInput(input, gradOutput)
	if gradOutput == nil or gradOutput:dim() == 0 then
		self.gradInput = nil
		return self.gradInput
	else
		return Parent.updateGradInput(self,input, gradOutput)
	end
end
