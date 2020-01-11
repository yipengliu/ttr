function [Xred,reg] = updateUred(tt, X, L,N, ind)
%Converts TT-tensor to two matrixs whose prodcut equals to the TT-matrix
%   [A]=updateU(tt,L) --- compose first L dimensions factors into a
%   matrix U
%
%---------------------------

 n=tt.n; ps=tt.ps; core=tt.core; r=tt.r;
 wp=n(ind)*r(ind)*r(ind+1);
%% compute Ured for ind=L
if ind==L
    if ind-1==1
        al = reshape(core(ps(1):ps(2)-1),[n(1),r(2)]) ;
    else
        al= core(ps(1):ps(2)-1);
        for i=2:ind-1
           cr=core(ps(i):ps(i+1)-1);
           cr=reshape(cr,[r(i),n(i)*r(i+1)]);
           al=reshape(al,[numel(al)/r(i),r(i)]);
           al=al*cr;
        end
        al=reshape(al,[numel(al)/r(ind),r(ind)]); 
    end
    Ured=al;
    reg=kron(Ured'*Ured,eye(n(ind),n(ind)));
    Xred=reshape(X*Ured,[N,n(ind)*r(ind)]);
    return;
end

%% comput Ured for ind=1
if ind==1
    if ind+1==L
        ar=reshape(core(ps(L):ps(L+1)-1),[r(L)*n(L),r(L+1)]);
    else
        ar=core(ps(L):ps(L+1)-1);
        for i=L-1:-1:ind+1
          cr=core(ps(i):ps(i+1)-1);
          cr=reshape(cr,[r(i)*n(i),r(i+1)]);
          ar=reshape(ar,[r(i+1),numel(ar)/r(i+1)]);
          ar=cr*ar;
        end
        ar=reshape(ar,[numel(ar)/r(L+1),r(L+1)]);  
    end
    ar=reshape(ar,[r(ind+1),numel(ar)/(r(ind+1)*r(L+1)),r(L+1)]);
    ar=permute(ar,[2,1,3]);
    Ured=permute(ar,[1,3,2]);
    Ured=reshape(Ured,[numel(ar)/(r(ind+1)),(r(ind+1))]);
    reg=kron(Ured'*Ured,eye(n(ind),n(ind)));
    Xred=zeros(N,r(L+1),wp);
for l =1:r(L+1)
    
    Xred(:,l,:)=reshape(X*ar(:,:,l),[N,wp]);
end
    return;
end
 %% compute Ured for ind~=1 and ind~=L   
 %compute left factors
if ind-1==1
    al = reshape(core(ps(1):ps(2)-1),[n(1),r(2)]) ;
else
    al= core(ps(1):ps(2)-1);
    for i=2:ind-1
       cr=core(ps(i):ps(i+1)-1);
       cr=reshape(cr,[r(i),n(i)*r(i+1)]);
       al=reshape(al,[numel(al)/r(i),r(i)]);
       al=al*cr;
    end
    al=reshape(al,[numel(al)/r(ind),r(ind)]); 
end

%compute right factors
if ind+1==L
    ar=reshape(core(ps(L):ps(L+1)-1),[r(L)*n(L),r(L+1)]);
else
    ar=core(ps(L):ps(L+1)-1);
    for i=L-1:-1:ind+1
      cr=core(ps(i):ps(i+1)-1);
      cr=reshape(cr,[r(i)*n(i),r(i+1)]);
      ar=reshape(ar,[r(i+1),numel(ar)/r(i+1)]);
      ar=cr*ar;
    end
    ar=reshape(ar,[numel(ar)/r(L+1),r(L+1)]);  
end

Xred=zeros(N,r(L+1),wp);
Ured=zeros(numel(al)*numel(ar)/(r(ind+1)*r(ind)*r(L+1)),r(L+1),(r(ind+1)*r(ind)));
for l =1:r(L+1)
    att=ar(:,l)*reshape(al,[1,numel(al)]);
    att=permute(reshape(att,[r(ind+1),numel(att)/(r(ind+1)*r(ind)),r(ind)]),[2,3,1]);
    att=reshape(att,[numel(att)/(r(ind+1)*r(ind)),(r(ind+1)*r(ind))]);
    Ured(:,l,:)=att;
    Xred(:,l,:)=reshape(X*att,[N,wp]);
end
Ured=reshape(Ured,[numel(al)*numel(ar)/(r(ind+1)*r(ind)),(r(ind+1)*r(ind))]);
reg=kron(Ured'*Ured,eye(n(ind),n(ind)));
return;