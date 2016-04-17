function [zsAcc, zsAcc_norm] = zeroshotDAP(yZsTe, TeC, probTe, prior, attB)    
     
pred = zeros(size(TeC, 2), size(probTe, 2)) ;

for j = 1:size(TeC, 2)
    %Debug: prA_C = ones(1,85) ;
    prA_C = prior ;
    prA_C(attB(TeC(j),:) == 0) = 1 - prA_C(attB(TeC(j),:) == 0) ;
    temp = probTe ;
    temp(attB(TeC(j),:) == 0,:) = 1- temp(attB(TeC(j),:) == 0, :) ;

    pp = bsxfun(@rdivide, temp, prA_C') ;
%     pred(j,:) = prod(pp) ;
    pred(j,:) = sum(pp) ;

end

[~, y] = max(pred) ;
confusionMatrix = zeros(size(TeC, 2), size(TeC, 2)) ;
for ii = 1:size(TeC, 2)
    idx = (yZsTe == ii) ;
    confusionMatrix(ii,:) = hist(y(idx), 1:size(TeC, 2)) ;
end
nCM = bsxfun(@rdivide, confusionMatrix, sum(confusionMatrix, 2)) ;
zsAcc_norm = mean(diag(nCM)) ;
zsAcc = mean(y == yZsTe') ;  

end