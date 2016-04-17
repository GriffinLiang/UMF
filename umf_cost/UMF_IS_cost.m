function [cost, grad] = UMF_IS_cost(data, label, U, V, W, SM_theta, params, weight)

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
numClasses = size(unique(category_label(:)'), 2);
numCases = size(data, 2);
groundTruth = full(sparse(category_label, 1:numCases, 1));

U = reshape(U, NrF, numClasses);
V = reshape(V, NrF, NrA);
W = reshape(W, NrF, NrD);
SM_theta = reshape(SM_theta, numClasses, NrD);
lambda1 = params.lambda1;
lambda2 = params.lambda2;
a = params.a ;

if(params.mode == 1)
    grad = zeros(size(U)) ;
elseif(params.mode == 2)
    grad = zeros(size(V)) ;
elseif(params.mode == 3)
    grad = zeros(size(W)) ;
elseif(params.mode == 4)
    grad = zeros(size(SM_theta)) ;
end

M = SM_theta*data ;
M = bsxfun(@minus, M, max(M, [], 1)) ;
category_prediction = bsxfun(@rdivide, exp(M), sum(exp(M)));
f = sigmoid(V'*((U*category_prediction).*(W*data))) ;

if (exist('weight', 'var'))
    cost = cost - (1/size(data, 2))*sum(sum(weight.*(attribute_label.*log(f) + ...
                                    (1-attribute_label).*log(1 - f)))) ;

    if(params.mode == 1)
        grad = grad + (1/size(data, 2))*(W*data).*(V*(weight.*(f-attribute_label)))*category_prediction' ;
    elseif(params.mode == 2)
        grad = grad + (1/size(data, 2))*((U*category_prediction).*(W*data))*(weight.*(f-attribute_label))' ;        
    elseif(params.mode == 3)
        grad = grad + (1/size(data, 2))*(U*category_prediction).*(V*(weight.*(f-attribute_label)))*data' ;
    elseif(params.mode == 4)
        SM_cost = -sum(log(sum(groundTruth.*exp(M))./sum(exp(M))))/numCases ;
        cost = cost + a*SM_cost ;
        grad = grad - a*(groundTruth-category_prediction)*data' ;    
        C = (U'*((W*data).*(V*(weight.*(f-attribute_label))))) ;
        indic = eye(numClasses) ;
        for ii = 1:size(grad, 1)
            grad(ii,:) = grad(ii,:) + (sum(bsxfun(@minus, indic(ii,:), ...
                    category_prediction').*C', 2).*category_prediction(ii,:)')'*data' ;
        end   
    end
else
    cost = cost - (1/size(data, 2))*sum(sum(attribute_label.*log(f) + (1-attribute_label).*log(1 - f))) ;

    if(params.mode == 1)
        grad = grad + (1/size(data, 2))*(W*data).*(V*(f-attribute_label))*category_prediction' ;
    elseif(params.mode == 2)
        grad = grad + (1/size(data, 2))*((U*category_prediction).*(W*data))*(f-attribute_label)' ;        
    elseif(params.mode == 3)
            grad = grad + (1/size(data, 2))*(U*category_prediction).*(V*(f-attribute_label))*data' ;
    elseif(params.mode == 4)
        SM_cost = -sum(log(sum(groundTruth.*exp(M))./sum(exp(M))))/numCases ;
        cost = cost + a*SM_cost ;
        grad = grad - a*(groundTruth-category_prediction)*data' ;    
        C = (U'*((W*data).*(V*(f-attribute_label)))) ;
        indic = eye(numClasses) ;
        for ii = 1:size(grad, 1)
            grad(ii,:) = grad(ii,:) + (sum(bsxfun(@minus, indic(ii,:), ...
                    category_prediction').*C', 2).*category_prediction(ii,:)')'*data' ;
        end   
    end    
end

        
if(params.mode == 1)
    cost = cost + 0.5*lambda1*sum(sum(U.^2)) ;
    grad = grad + lambda1*U ;
elseif(params.mode == 2)
    cost = cost + 0.5*lambda1*sum(sum(V.^2)) ;
    grad = grad + lambda1*V ;
elseif(params.mode == 3)
    cost = cost + 0.5*lambda1*sum(sum(W.^2)) ;
    grad = grad + lambda1*W ;
elseif(params.mode == 4)
    cost = cost + 0.5*lambda2*sum(sum(SM_theta.^2)) ;
    grad = (1/numCases)*grad + lambda2*SM_theta;        
end

grad = grad(:);

if( useGpu == true)
    cost = gather(cost) ;
    grad = gather(grad) ;
end

end