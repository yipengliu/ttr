function  [result] = rmse(X,Y)
result=sqrt(sum((X(:)-Y(:)).^2)/numel(X));
end
