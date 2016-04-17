function [L] = processClaLab(C, nImClass)

    L = [] ;
    for iClass = 1:size(C, 2)
        c = C(iClass) ;        
        L = [L ; iClass*ones(nImClass(c), 1)] ;
    end
    
end