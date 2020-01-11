function [r]=mycorrcoef(y1,y2)

r = corrcoef(y1,y2);
r = r(1,2);