function [a] = updateU(tt, L)
%Converts TT-tensor to two matrixs whose prodcut equals to the TT-matrix
%   [A]=updateU(tt,L) --- compose first L dimensions factors into a
%   matrix U
%
%---------------------------

 n=tt.n; ps=tt.ps; core=tt.core; r=tt.r;
if L==1
    a=reshape(core(ps(1):ps(2)-1),[n(1),r(2)]);
else
    a=core(ps(1):ps(2)-1);
    for i=2:L
      cr=core(ps(i):ps(i+1)-1);
      cr=reshape(cr,[r(i),n(i)*r(i+1)]);
      a=reshape(a,[numel(a)/r(i),r(i)]);
      a=a*cr;
    end
    a=reshape(a,[prod(n(1:L)),r(L+1)]);   
end
return;