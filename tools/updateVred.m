function [a] = updateVred(tt, M, ind)
%Converts TT-tensor to two matrixs whose prodcut equals to the TT-matrix
%   [A]=updateU(tt,L) --- compose first L dimensions factors into a
%   matrix U
%
%---------------------------

d=tt.d; n=tt.n; ps=tt.ps; core=tt.core; r=tt.r; L=d-M;
%% compute Vred for ind=L+1
if ind==L+1
    if ind+1==d
        ar=reshape(core(ps(d):ps(d+1)-1),[r(d),n(d)*r(d+1)]);
    else
        ar=core(ps(d):ps(d+1)-1);
        for i=d-1:-1:ind+1
          cr=core(ps(i):ps(i+1)-1);
          cr=reshape(cr,[r(i)*n(i),r(i+1)]);
          ar=reshape(ar,[r(i+1),numel(ar)/r(i+1)]);
          ar=cr*ar;
        end
        ar=reshape(ar,[r(ind+1),numel(ar)/r(ind+1)]);  
    end
    a=ar';
    return;
end

%% compute Vred for ind=L+M
if ind==L+M
    if ind-1==L+1
        al = reshape(core(ps(L+1):ps(L+2)-1),[r(L+1),n(L+1),r(L+2)]) ;
    else
        al= core(ps(L+1):ps(L+2)-1);
        for i=L+2:ind-1
           cr=core(ps(i):ps(i+1)-1);
           cr=reshape(cr,[r(i),n(i)*r(i+1)]);
           al=reshape(al,[numel(al)/r(i),r(i)]);
           al=al*cr;
        end
        al=reshape(al,[r(L+1),numel(al)/(r(ind)*r(L+1)),r(ind)]); 
    end
    a=permute(al,[2,1,3]); 
    a=reshape(a,[numel(a)/(r(ind)*r(ind+1)),(r(ind)*r(ind+1))]);
    return;
end

%% compute Vred for ind~=L+1 and ind~=L+M
if ind+1==d
        ar=reshape(core(ps(d):ps(d+1)-1),[r(d),n(d)*r(d+1)]);
    else
        ar=core(ps(d):ps(d+1)-1);
        for i=d-1:-1:ind+1
          cr=core(ps(i):ps(i+1)-1);
          cr=reshape(cr,[r(i)*n(i),r(i+1)]);
          ar=reshape(ar,[r(i+1),numel(ar)/r(i+1)]);
          ar=cr*ar;
        end
        ar=reshape(ar,[r(ind+1),numel(ar)/r(ind+1)]);  
end
if ind-1==L+1
        al = reshape(core(ps(L+1):ps(L+2)-1),[r(L+1),n(L+1),r(L+2)]) ;
else
        al= core(ps(L+1):ps(L+2)-1);
        for i=L+2:ind-1
           cr=core(ps(i):ps(i+1)-1);
           cr=reshape(cr,[r(i),n(i)*r(i+1)]);
           al=reshape(al,[numel(al)/r(i),r(i)]);
           al=al*cr;
        end
        al=reshape(al,[r(L+1),numel(al)/(r(ind)*r(L+1)),r(ind)]); 
end
a=zeros(numel(ar)*numel(al)/(r(ind+1)*r(ind)*r(L+1)),r(L+1),(r(ind+1)*r(ind)));
for l=1:r(L+1)
    att=reshape(ar,[numel(ar),1])*reshape(al(l,:,:),[1,numel(al(l,:,:))]);
    att=permute(reshape(att,[r(ind+1),numel(att)/(r(ind+1)*r(ind)),r(ind)]),[2,3,1]);
    a(:,l,:)=reshape(att,[numel(att)/(r(ind+1)*r(ind)),(r(ind+1)*r(ind))]);
end
  a=reshape(a,[numel(a)/(r(ind)*r(ind+1)),(r(ind)*r(ind+1))]);  
return