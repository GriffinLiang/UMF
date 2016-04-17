function [cost, grad] = UMF_CS_cost(data, label, U, V, W, params, weight)

global useGpu ;

if( useGpu == true)
    data = gpuArray( data );
end

attribute_label = label.attribute ;
attribute_label(attribute_label == -1) = 0 ;
category_label = label.category ;

cost = 0 ;
NrA = size(attribute_label, 1) ;
NrD = size(data, 1);
NrF = params.NrF;
numCases = size(data, 2);

if(size(category_label, 1) == 1)
    numClasses = size(unique(category_label(:)'), 2);
    category_groundTruth = full(sparse(category_label, 1:numCases, 1));
else
    category_groundTruth = category_label;
    numClasses = size(category_label, 1);
end

U = reshape(U, NrF, numClasses);
V = reshape(V, NrF, NrA);
W = reshape(W, NrF, NrD);
lambda = params.lambda1 ;

if(params.mode == 1)
    grad = zeros(size(U)) ;
elseif(params.mode == 2)
    grad = zeros(size(V)) ;
elseif(params.mode == 3)
    grad = zeros(size(W)) ;
end


f = sigmoid(V'*((U*category_groundTruth).*(W*data))) ;

if (exist('weight', 'var'))
    cost = cost - (1/size(data, 2))*sum(sum(weight.*(attribute_label.*log(f) + ...
                                    (1-attribute_label).*log(1 - f)))) ;

    if(params.mode == 1)
        grad = grad + (1/size(data, 2))*(W*data).*(V*(weight.*(f-attribute_label)))*category_groundTruth' ;
    elseif(params.mode == 2)
        grad = grad + (1/size(data, 2))*((U*category_groundTruth).*(W*data))*(weight.*(f-attribute_label))' ;        
    elseif(params.mode == 3)
        grad = grad + (1/size(data, 2))*(U*category_groundTruth).*(V*(weight.*(f-attribute_label)))*data' ;
    end
else
    cost = cost - (1/size(data, 2))*sum(sum(attribute_label.*log(f) + (1-attribute_label).*log(1 - f))) ;

    if(params.mode == 1)
        grad = grad + (1/size(data, 2))*(W*data).*(V*(f-attribute_label))*category_groundTruth' ;
    elseif(params.mode == 2)
        grad = grad + (1/size(data, 2))*((U*category_groundTruth).*(W*data))*(f-attribute_label)' ;        
    elseif(params.mode == 3)
        grad = grad + (1/size(data, 2))*(U*category_groundTruth).*(V*(f-attribute_label))*data' ;
    end  
end

if (exist('lambda', 'var'))         
    if(params.mode == 1)
        cost = cost + 0.5*lambda*sum(sum(U.^2)) ;
        grad = grad + lambda*U ;
    elseif(params.mode == 2)
        cost = cost + 0.5*lambda*sum(sum(V.^2)) ;
        grad = grad + lambda*V ;
    elseif(params.mode == 3)
        cost = cost + 0.5*lambda*sum(sum(W.^2)) ;
        grad = grad + lambda*W ;    
    end    
end

grad = grad(:);

if( useGpu == true)
    cost = gather(cost) ;
    grad = gather(grad) ;
end

end