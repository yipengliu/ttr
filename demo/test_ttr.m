%simulation of comparison of the TTR and CP_based tensor on tensor Regression
clear all;
%run simulation for predictive performance 
para.P = [15,20];
para.Q = [5,10];
para.L = length(para.P);
para.M = length(para.Q);
para.dim = [para.P,para.Q];
para.maxiter=1000;
para.datarep=20;
% para.testn=200;
Rlist={[1,3,2,2,1],[1,5,3,3,1]};
samplelist={100:20:280,140:20:320};
s2nlevel=2;
Rnum=2;
samplenum=10;
repout_ttr=zeros(s2nlevel,Rnum,samplenum,para.datarep);
predrmse_ttr=zeros(s2nlevel,Rnum,samplenum,para.datarep);
cor_ttr=zeros(s2nlevel,Rnum,samplenum,para.datarep);
Q2_ttr=zeros(s2nlevel,Rnum,samplenum,para.datarep);
runtime_ttr=zeros(s2nlevel,Rnum,samplenum,para.datarep);
for s=1%:2
    s2n=10*s;
for R_ind = 1:2
%      override=true;
%      [returnStatus]=genaratedataset(para, Rlist{R_ind},R_ind,samplelist{R_ind});
    for n_ind=1:10
        n=samplelist{R_ind}(n_ind);   
        for i=1%:para.datarep
             %load data
            load(['data/trainx_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'trainx');
            load(['data/trainy_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'trainy');
            load(['data/valx_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'valx');
            load(['data/valy_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'valy');
            load(['data/testx_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'testx');
            load(['data/testy_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'testy');
            % Train the model.
           load(['data/model_',num2str(R_ind),'.mat'], 'origin_model');
            para.N=n;
            % training method
%             best_mormse=100;
%             for l=-5:2
            para.lambda = 10^(-2);
            [model,runtime_ttr(s,R_ind,n_ind,i)] = ttr(para, Rlist{R_ind}, trainx,trainy,valx,valy);
            %est_model and error
            est_model=full(model);
%             mormse=rmse(est_model,origin_model);
%             if mormse<=best_mormse
%                 best_mormse=mormse;
%                 best_model=est_model;
%             end
%             end
            repout_ttr(s,R_ind,n_ind,i)=rmse(est_model,origin_model);
            % est_testy and error
            est_testy=contract(testx,est_model,para.L);
%             rmse(zscore(reshape(est_testy,[numel(testy),1])),zscore(reshape(testy,[numel(testy),1])));
            Ypred=zscore(reshape(est_testy,[numel(testy),1]));
            Y=zscore(reshape(testy,[numel(testy),1]));
            cor_ttr(s,R_ind,n_ind,i) = mycorrcoef(Ypred(:),Y(:));
            Ypress = sum((Y(:)-Ypred(:)).^2);
            predrmse_ttr(s,R_ind,n_ind,i)  = sqrt(Ypress./numel(Y));
            Q2_ttr(s,R_ind,n_ind,i) = 1 - Ypress./sum(Y(:).^2);
%             predrmse_ttr(s,R_ind,n_ind,i)=rmse(zscore(reshape(est_testy,[numel(testy),1])),zscore(reshape(testy,[numel(testy),1])));
        end
    end
end
end
% repout_ttr=reshape(repout_ttr,[s2nlevel*Rnum*samplenum,para.datarep]);
% modelrmse=reshape(mean(repout_ttr,2),[s2nlevel,Rnum,samplenum]);
% save('result/model_rmse_ttr.mat', 'modelrmse');
% 
% runtime_ttr=reshape(runtime_ttr,[s2nlevel*Rnum*samplenum,para.datarep]);
% runtime=reshape(mean(runtime_ttr,2),[s2nlevel,Rnum,samplenum]);
% save('result/runtime_ttr.mat', 'runtime');
% 
%  predrmse_ttr=reshape( predrmse_ttr,[s2nlevel*Rnum*samplenum,para.datarep]);
%  predrmse=reshape(mean( predrmse_ttr,2),[s2nlevel,Rnum,samplenum]);
% save('result/pred_rmse_ttr.mat', 'predrmse');
% 
% cor_ttr=reshape( cor_ttr,[s2nlevel*Rnum*samplenum,para.datarep]);
%  cor=reshape(mean( cor_ttr,2),[s2nlevel,Rnum,samplenum]);
% save('result/pred_cor_ttr.mat', 'cor');
% 
% Q2_ttr=reshape( Q2_ttr,[s2nlevel*Rnum*samplenum,para.datarep]);
% Q2=reshape(mean( Q2_ttr,2),[s2nlevel,Rnum,samplenum]);
% save('result/pred_Q2_ttr.mat', 'Q2');