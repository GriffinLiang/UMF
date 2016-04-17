function auc = computeAUC(scores, labels)

nAtt = size(labels, 1) ;
auc = zeros(1, nAtt) ;

for iAtt = 1:nAtt 
    if(size(unique(labels(iAtt,:)), 2) == 1)
        continue
    end
    
    roc = computeROC(scores(iAtt,:)', labels(iAtt,:)') ;       
	auc(iAtt) = roc.area ;
     
end

