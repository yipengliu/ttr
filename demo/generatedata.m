function[X,Y,s2n_scale]=generatedata(para,N ,U,V)

P=para.P;
Q=para.Q;
noiselevel=para.noiselevel;

Ye = random('Normal', 0, 1, [N,Q]);
X = random('Normal', 0, 1, [N,P]);
Xmat =reshape(X,[N,prod(P)]);
Ysig = Xmat*U*V;
Ysig =reshape(Ysig,[N,Q]);
% s2n_scale = s2n*sqrt(sum(Ye(:).^2))/sqrt(sum(Ysig(:).^2));
Y = Ysig+noiselevel.*Ye;

end