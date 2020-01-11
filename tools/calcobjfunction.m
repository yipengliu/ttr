function [ funcValue ] = calcobjfunction(para, model, X, Y )
%CalcObjFunc Calculate the objective function by the given models, lambda and X,Y.
%   Parameters:
%       para: some parameters.
%       model: cofficient array in TT-format. 
%       lambda: The coefficient of the penalty item.
%       X: predictors
%       Y: responses 
%
%   Formula: funcValue = \sum(Y - <model , X>_L)^2 + \lambda *
%   \sum(model)^2
%
%Tensor Train Regression
%Copyright 2017
%

% compose tt-tensor to full tensor
modelTensor = full(model);

estimated_Y = contract(X,modelTensor,para.L);

funcValuePart1 = sum(reshape((Y-estimated_Y),[numel(Y),1]).^2);
% funcValuePart1 = funcValuePart1 / datasetSize;

funcValuePart2 = norm(reshape(modelTensor,[prod(para.P),prod(para.Q)]),'fro')^2;


funcValue = funcValuePart1 + para.lambda * funcValuePart2;

end

