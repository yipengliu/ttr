function [a] = updateV(tt, M)
%Converts TT-tensor to two matrixs whose prodcut equals to the TT-matrix
%   [A]=updateU(tt,M) --- compose last M dimensions factors into a
%   matrix V
%
%---------------------------

d=tt.d; n=tt.n; ps=tt.ps; core=tt.core; r=tt.r;
if M==1
    a=reshape(core(ps(d):ps(d+1)-1),[r(d),n(d)]);
else
    a=core(ps(d):ps(d+1)-1);
    for i=d-1:-1:d-M+1
      cr=core(ps(i):ps(i+1)-1);
      cr=reshape(cr,[r(i)*n(i),r(i+1)]);
      a=reshape(a,[r(i+1),numel(a)/r(i+1)]);
      a=cr*a;
    end
    a=reshape(a,[r(d-M+1),numel(a)/r(d-M+1)]);  
end
return;