function[Y]=contract(X,model,L)
% compute the contract product between two tensors.
%    Y= <X,model>_L

dimx=size(X);
N=dimx(1);
P=dimx(2:L+1);
dimmodel=size(model);
Q=dimmodel(L+1:length(dimmodel));
Xmat=reshape(X,[N,prod(P)]);
modelmat=reshape(model,[prod(P),prod(Q)]);
Y=Xmat*modelmat;
dimy=[N,Q];
if length(dimy)>2
Y=reshape(Y,dimy);
end
