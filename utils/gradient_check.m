
%% Gradient 
nData = 10 ;
nF = 10 ;
nAtt = 20 ;
nDim = 30 ;
lambda = 0.1 ;
GDT_data = rand(nDim,nData) ;
GDT_attribute_label = randn(nAtt, nData) > 0 ;
% GDT_attribute_label = GDT_attribute_label*2 - 1 ;
% GDT_attribute_label(randn(nAtt, nData) > 0) = 0 ;
GDT_category_label = (randn(1, nData) > 0) + 1 ;
% GDT_category_label = sparse(GDT_category_label, 1:size(GDT_data, 2), 1) ;
% GDT_weight = zeros(size(GDT_attribute_label));
% for ii = 1:size(GDT_attribute_label, 1)
%     pos_num = sum(GDT_attribute_label(ii, :) == 1);
%     neg_num = sum(GDT_attribute_label(ii, :) == 0);
%     GDT_weight(ii, GDT_attribute_label(ii, :)==1) = (pos_num + neg_num)/(2*pos_num);
%     GDT_weight(ii, GDT_attribute_label(ii, :)==0) = (pos_num + neg_num)/(2*neg_num);
% end

GDT_data = GDT_data ;
GDT_label.attribute = GDT_attribute_label ;
GDT_label.category = GDT_category_label ;
GDT_U = initializeParameters(nF, 2);
GDT_V = initializeParameters(nF, nDim) ;
GDT_W = initializeParameters(nF, nDim) ;
GDT_LR = initializeParameters(nAtt, nF);

GDT_params.NrF = nF;
GDT_params.lambda_u = lambda ;
GDT_params.lambda_v = lambda ;
GDT_params.lambda_w = lambda ;
GDT_params.lambda_sm = lambda ;
GDT_params.a = 0.2;
GDT_params.mode = 4;
GDT_params.Sigm = 0;

[cost, grad] = UMF_IC_H_Alter_cost(GDT_data, GDT_label, GDT_U(:), GDT_V(:), GDT_W(:), ...
                                GDT_LR(:), GDT_params) ;
% Check gradients
numGrad = computeNumericalGradient( @(p) UMF_IC_H_Alter_cost(GDT_data, GDT_label, GDT_U(:), ...
                  GDT_V(:), GDT_W(:), p, GDT_params), GDT_LR(:));


% Use this to visually compare the gradients side by side
disp([numGrad grad]); 

diff = norm(numGrad-grad)/norm(numGrad+grad);
% Should be small. In our implementation, these values are usually less than 1e-9.
disp(diff); 

assert(diff < 1e-9, 'Difference too large. Check your gradient computation again');