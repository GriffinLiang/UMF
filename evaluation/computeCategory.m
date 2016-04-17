function [acc, value] = computeCategory(theta, data, label)

M = theta*data ;
M = bsxfun(@minus, M, max(M));
value = bsxfun(@rdivide, exp(M), sum(exp(M)));
[~, pred] = max(value) ;
acc = mean(pred == label) ;