function [cost, grad] = logRegCost(theta, data, labels, lambda, weight)
%% label in {0,1}, 

global useGpu ;

if( useGpu == true)
    data = gpuArray( data );
end

theta = reshape(theta, size(data, 1), size(labels, 1)) ;
cost = 0 ;
grad = zeros(size(theta)) ;
h = sigmoid(theta'*data) ;

if (exist('weight', 'var'))
    cost = cost - (1/size(data, 2))*sum(sum(weight.*(labels.*log(h) + ...
                                    (1-labels).*log(1 - h)))) ;
    grad = grad + (1/size(data, 2))*data*(weight.*(h-labels))' ;
else
    cost = cost - (1/size(data, 2))*sum(sum(labels.*log(h) + (1-labels).*log(1 - h))) ;
    grad = grad + (1/size(data, 2))*data*(h-labels)' ;
end

if (exist('lambda', 'var'))
    cost = cost + 0.5*lambda*sum(sum(theta.^2)) ;
    grad = grad + lambda*theta ;
end

grad = grad(:) ;

if( useGpu == true)
    cost = gather(cost) ;
    grad = gather(grad) ;
end

end
