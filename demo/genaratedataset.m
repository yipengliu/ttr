function[returnStatus]=genaratedataset(para, rank,R_ind,samplelist)
    

% if override == false && ...
%     exist('data2/trainx.mat', 'file') ~= 0 && ...
%     exist('data2/valx.mat', 'file') ~= 0 && ...
%     exist('data2/testx.mat', 'file') ~= 0
% 
%     disp('Generate the datasets. Skip.');
%     return;
% end
%% parameters
L=para.L;M=para.M;
datarep=para.datarep;
%% model
model = initmodel(para,rank);
U = updateU(model,L);
V = updateV(model,M);
origin_model=full(model);
save(['data2/model_',num2str(R_ind),'.mat'],'origin_model');
for s=1:2
    s2n=10*s;
%     para.noiselevel=1/((10)^(s));
for n = samplelist
    for i =1:datarep
        %% train data
        [trainx,trainy,s2nscale]=generatedata_train(para,n,U,V,s2n);
        save(['data2/trainx_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'trainx');
        save(['data2/trainy_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'trainy');
        
        %% val data
        [valx,valy]=generatedata_test(para,n,U,V,s2nscale);
        save(['data2/valx_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'valx');
        save(['data2/valy_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'valy');
        
        %% test data
        [testx,testy]=generatedata_test(para,n,U,V,s2nscale);
        save(['data2/testx_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'testx');
        save(['data2/testy_',num2str(R_ind),'_',num2str(s2n),'_',num2str(n),'_',num2str(i),'.mat'], 'testy');
    end
end
end
%% finish
disp('Generate the datasets. Finish.');
returnStatus = true;
end