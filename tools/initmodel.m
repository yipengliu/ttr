function [model] = initmodel(para, rank )
%InitModels Initialize the models by arguments.
%   Parameters:
%       responseNum: The number of responses.
%       D_way: The number of components.
%       dims: A array of dims of each component.
%       rank: The rank of tensor decomposition and composition.
%
%   Initialize the components of model by random values.
%
%Sparse Tensor Regression
%Copyright 2017, Space Liang. Email: root [at] lyq.me
%

model = tt_random(para.dim,para.M+para.L,rank)

end

