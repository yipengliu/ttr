function [ model,runtime,trainerror ] = ttr( para, rank, X,Y, Xv, Yv )
%  Tensor Train Regression
%   Parameters:
%       lambda: The coefficent of the penalty term.
%       rank: A vector representes rank of Tensor Train decomposition.
%       trainx: predictor in train set.
%       valx: predictor in valset to avoid overfitting.
%       trainy: output in trainset.
%       valy: output in valset to avoid overfitting.
%
%   Formula: funcValue = \sum(Y - <model , X>_L)^2 + \lambda *
%   \sum(model)^2
%   Given Y, X and lambda.
%   We let the partial derivative to be zero and update the model B by its
%   closed-form solution.
%
%   Tensor Train Regression
%% set parameters
iterStart = 0;
iterTotal = para.maxiter;
shakyRate = 1.5;
N=para.N;
L=para.L;
M=para.M;
P=para.P;
Q=para.Q;
lambda=para.lambda;

%% initialization

%     initialize the random model
model = initmodel(para,rank);

%% permutation for X and Y
%X permutation
if L~=1
     Xarrays=cell(1,L);
     for i =1:L
        index=[1,i+1:1:L+1,2:1:i];
%         index=circshift(1:1:L,[0,1-i])+1;
        Xarrays{i}= reshape(permute(X,index),[N,P(i),prod(P)/P(i)]);
    end
end
Xmat = reshape(X,[N,prod(P)]);
%Y permutation
if M~=1
    Yarrays=cell(1,M);
    for j =1:M
        index=[1,j+2:1:M+1,2:1:j,j+1];
%         index=circshift(1:1:M,[0,-j])+1;
        Yarrays{j}= reshape(permute(Y,index),[N,prod(Q)/Q(j),Q(j)]);
    end
end
Ymat = reshape(Y,[N,prod(Q)]);



%% update U ,V
V = updateV(model,M);



%% 
minTrainingFuncValue = calcobjfunction(para,model, X,Y);
minValidationFuncValue = calcobjfunction(para,model,Xv,Yv);

disp('Train the model. Running...');
tic;

%% main loop
% trainerror=zeros();
for iter = iterStart+1:iterTotal
    
    %% update first L dimensions
   d=model.d; n=model.n; ps=model.ps;  r=model.r;
    for i = 1:L 
    if L==1
        Ul=(Xmat'*Xmat+lambda*eye(n(i),n(i)))\(Xmat'*(Ymat*V'*pinv(V*V')));
        model.core(ps(1):ps(2)-1)=reshape(permute(reshape(Ul,[n(i),r(i),r(i+1)]),[2,1,3]),[numel(Ul),1]);
%         calcobjfunction(para,model,X ,Y)
    else
        if i==L
                Xarrayeach=reshape(Xarrays{i},[N*n(i),prod(P)/n(i)]);
                [C,reg]=updateUred(model,Xarrayeach,L,N,i);
                Ul=(C'*C+lambda.* (reg))\(C'*(Ymat*V'*pinv(V*V')));
                model.core(ps(i):ps(i+1)-1)=reshape(permute(reshape(Ul,[n(i),r(i),r(i+1)]),[2,1,3]),[numel(Ul),1]);
%                 calcobjfunction(para,model,X ,Y)
        else
                wp=n(i)*r(i)*r(i+1);
                Xarrayeach=reshape(Xarrays{i},[N*n(i),prod(P)/n(i)]);
                [C,reg]=updateUred(model,Xarrayeach,L,N,i);
                C=reshape(C,[N*r(L+1),wp]);
                Ul=(C'*C+lambda.* (reg))\(C'*reshape(Ymat*V'*pinv(V*V'),[N*r(L+1),1]));
                model.core(ps(i):ps(i+1)-1)=reshape(permute(reshape(Ul,[n(i),r(i),r(i+1)]),[2,1,3]),[numel(Ul),1]);
%                 calcobjfunction(para,model,X ,Y)
        end
    end
    end
    U = updateU(model,L);
    Xred=Xmat*U;
    Xred_inv=((Xred)'*(Xred)+lambda*(U'*U))\(Xred)';
    %% update last M dimensions
    for i = L+1:d
        if M==1 
           V=Xred_inv*Ymat;
           model.core(ps(L+1):ps(L+2)-1)=reshape(V,[numel(V),1]);
%            calcobjfunction(para,model,X ,Y)
        else
           if i==L+1
                D=updateVred(model,M,i);
                mat=Xred_inv*reshape(Yarrays{i-L},[N,prod(Q)]);
                mat=reshape(permute(reshape(mat,[r(L+1),numel(mat)/(r(L+1)*n(i)),n(i)]),[2,3,1]),[numel(mat)/(r(L+1)*n(i)),(r(L+1)*n(i))]);
                Vm=(D'*D)\(D'*mat);
                model.core(ps(i):ps(i+1)-1)=reshape(permute(reshape(Vm,[r(i+1),n(i),r(i)]),[3,2,1]),[numel(Vm),1]);
%                 calcobjfunction(para,model,X ,Y)
           else
                D=updateVred(model,M,i);
                mat=Xred_inv*reshape(Yarrays{i-L},[N,prod(Q)]);
                mat=reshape(permute(reshape(mat,[r(L+1),numel(mat)/(r(L+1)*n(i)),n(i)]),[2,1,3]),[numel(mat)/n(i),n(i)]);
                Vm=(D'*D)\(D'*mat);
                model.core(ps(i):ps(i+1)-1)=reshape(permute(reshape(Vm,[r(i),r(i+1),n(i)]),[1,3,2]),[numel(Vm),1]);
%                 calcobjfunction(para,model,X ,Y)
           end
        end
    end
    
    %% compute error
   model=round(model,0.0001);
   V = updateV(model,M); 
   if norm(V,'fro')==0
      disp('error, too large lambda');
      break;  
   end
   
    trainingFuncValue = calcobjfunction(para,model,X ,Y);
    trainerror(iter)=trainingFuncValue;
    if abs(trainingFuncValue - minTrainingFuncValue)/minTrainingFuncValue<=1e-3
        break;
    end      
    if trainingFuncValue <= shakyRate * minTrainingFuncValue
        minTrainingFuncValue = min(minTrainingFuncValue, trainingFuncValue);
        disp('descening');
    else
        disp('not descening, error');
        break;
    end
       
    validationFuncValue = calcobjfunction(para,model,Xv,Yv);
    
    disp(['    Iter: ', num2str(iter), '. Training: ', num2str(trainingFuncValue), '. Validation: ', num2str(validationFuncValue)]);
    
    minValidationFuncValue = min(minValidationFuncValue, validationFuncValue);
   
    
end
%% save
% est_modeltensor=full(model);
% save('data/est_modeltensor.mat', 'est_modeltensor');
%%
runtime=toc;
disp('Train the model. Finish.');



end