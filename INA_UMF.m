clear;
load('imagenet_attribute_25_BB_DeCAF.mat') ;
load('attrann.mat') ;

category_label = repmat(1:384, 25, 1) ;      
category_label = category_label(:)' ;
attribute_label = attrann.labels' ;
attribute_label(attribute_label == 0) = 0.5 ;
attribute_label(attribute_label == -1) = 0 ;

feaTrain = bsxfun(@rdivide, feaTrain, sqrt(sum(feaTrain.^2))) ;
train_data = feaTrain(:, 1:3:end) ;
val_data = feaTrain(:, 2:3:end) ;
test_data = feaTrain(:, 3:3:end) ;
train_category_label = category_label(1:3:end) ;
val_category_label = category_label(2:3:end) ;
test_category_label = category_label(3:3:end) ;
train_attribute_labels = attribute_label(:, 1:3:end) ;
val_attribute_labels = attribute_label(:, 2:3:end) ;
test_attribute_labels = attribute_label(:, 3:3:end) ;
clear feaTrain category_label attribute_label attrann

% global useGpu
% useGpu = true;

%% Initialization
NrA = 25 ; 
NrD = size(train_data, 1) ;
NrF = 25 ;

fid = 1;
% fid = fopen('INA_UMF.txt', 'w');

%% Pretrain stage 1
%%%% optimization
options.Method = 'L-BFGS'; 
options.maxIter = 200 ;	
options.display = 'on'; 
lambda_pre1 = 1e-4 ;
% Train Stage
w_pre_attribute = initializeParameters(NrD, NrA);
[Opt_w_pre_attribute, cost] = minFunc( @(p) logRegCost(p, train_data, ...
            train_attribute_labels, lambda_pre1), w_pre_attribute(:), options);        
% Val stage
Opt_w_pre_attribute = reshape(Opt_w_pre_attribute, size(train_data, 1), NrA) ;
pred_value_val = sigmoid(Opt_w_pre_attribute'*val_data) ;
pred_binary_val = (pred_value_val >=0.5) ;
acc_val = mean(pred_binary_val == val_attribute_labels, 2) ;
auc_val = computeAUC(pred_value_val, val_attribute_labels) ;
fprintf(fid, 'Att Val mACC:%0.4f\tmAUC:%0.4f\n', mean(acc_val), mean(auc_val)) ; 
% Test Stage
pred_value_test = sigmoid(Opt_w_pre_attribute'*test_data) ;
pred_binary_test = (pred_value_test >=0.5) ;
acc_test = mean(pred_binary_test == test_attribute_labels, 2) ;
auc_test = computeAUC(pred_value_test, test_attribute_labels) ;
fprintf(fid, 'Att Test mACC:%0.4f\tmAUC:%0.4f\n', mean(acc_test), mean(auc_test)) ; 

% Pretrain stage SVD
[U_pre1,S_pre1,V_pre1] = svd(Opt_w_pre_attribute) ;
Opt_pre1_W = U_pre1(:, 1:NrF)*S_pre1(1:NrF, 1:NrF)^(0.5) ;
Opt_pre1_W = Opt_pre1_W' ; 
Opt_pre1_V = S_pre1(1:NrF, 1:NrF)^(0.5)*V_pre1(:, 1:NrF)' ;
pred_value_test_svd = sigmoid(Opt_pre1_V'*Opt_pre1_W*test_data) ;
pred_binary_test_svd = (pred_value_test_svd >=0.5) ;
acc_test_svd = mean(pred_binary_test_svd == test_attribute_labels, 2) ;
auc_test_svd = computeAUC(pred_value_test_svd, test_attribute_labels) ;
fprintf(fid, 'Att SVD Test mACC:%0.4f\tmAUC:%0.4f\n', mean(acc_test_svd), mean(auc_test_svd)) ; 


%% Pretrain Stage 2
options.Method = 'L-BFGS'; 
options.maxIter = 200 ;	
options.display = 'on'; 
numClasses = max(train_category_label) ;
lambda_pre2 = 1e-5 ;

SoftmaxTheta = initializeParameters(NrD, numClasses);
[SoftmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, numClasses, NrD, ...
                        lambda_pre2, train_data, train_category_label), ...                                   
                        SoftmaxTheta(:), options);                         
SoftmaxOptTheta = reshape(SoftmaxOptTheta, numClasses, size(train_data, 1));   
[train_acc, train_c_value] = computeCategory(SoftmaxOptTheta, train_data, ...
                             train_category_label) ;
[val_acc, val_c_value] = computeCategory(SoftmaxOptTheta, val_data, ...
                             val_category_label) ;
[test_acc, test_c_value] = computeCategory(SoftmaxOptTheta, test_data, ...
                             test_category_label) ;
fprintf(fid, 'Classification accuracy(tr,val,te): %0.2f%%\t%0.2f%%\t%0.2f%%\n', ...
         train_acc*100, val_acc*100, test_acc*100) ;
    
%% UMF
MaxIter = 50;
options.display = 'off'; 
options.maxIter = 10;
numClasses = max(train_category_label) ;
a = 0.5;
lambda = 10.^(-4);
Opt_W = Opt_pre1_W; 
Opt_V = Opt_pre1_V;
Opt_U = initializeParameters(NrF, numClasses);
Opt_SM_theta = SoftmaxOptTheta;

params.NrF = NrF;
params.lambda1 = lambda;
params.lambda2 = 0.1*lambda;
params.a = a;
params.mode = 1;
label.attribute = train_attribute_labels;
label.category = train_category_label;
mAUC_val_old = 0;
mAUC_test_old = 0;
num_cum = 0;
fprintf(fid, 'NrF:%d\t lamb_one:%f\t lamb_two:%f\t a:%0.2f\n', ...
              NrF, params.lambda1, params.lambda2, params.a);
fprintf(fid, 'Iter\t val_old\t val_new\t test\t test_new\n');
for ii = 1:MaxIter      
    Opt_W_old = Opt_W; 
    Opt_V_old = Opt_V;
    Opt_U_old = Opt_U;
    Opt_SM_theta_old = Opt_SM_theta;
    % Optimize U
    if(params.mode == 1)
    [Opt_U, cost] = minFunc( @(p) UMF_IS_cost(train_data, label, p, Opt_V(:), ...
                          Opt_W(:), Opt_SM_theta(:), params), Opt_U(:), options);
    end

    % Optimize V
    if(params.mode == 2)
    [Opt_V, cost] = minFunc( @(p) UMF_IS_cost(train_data, label, Opt_U(:), p, ...
                          Opt_W(:), Opt_SM_theta(:), params), Opt_V(:), options);
    end

    % Optimize W
    if(params.mode == 3)
    [Opt_W, cost] = minFunc( @(p) UMF_IS_cost(train_data, label, Opt_U(:), Opt_V(:), ...
                             p, Opt_SM_theta(:), params), Opt_W(:), options);
    end

    % Optimize SM
    if(params.mode == 4)
    [Opt_SM_theta, cost] = minFunc( @(p) UMF_IS_cost(train_data, label, Opt_U(:), ...
                        Opt_V(:), Opt_W(:), p, params), Opt_SM_theta(:), options);
    end

    U = reshape(Opt_U, NrF, numClasses);
    V = reshape(Opt_V, NrF, NrA);
    W = reshape(Opt_W, NrF, NrD);
    SM_theta = reshape(Opt_SM_theta, numClasses, NrD);

    [val_acc, p_val] = computeCategory(SM_theta, val_data, val_category_label) ;
    pred_val = sigmoid(V'*((U*p_val).*(W*val_data)));
    mAUC_val = mean(computeAUC(pred_val, val_attribute_labels));    
    fprintf(fid, 'Iter%d\t%0.4f\t%0.4f\t',ii, mAUC_val_old, mAUC_val);

    [test_acc, p_test] = computeCategory(SM_theta, test_data, test_category_label) ;
    pred_test = sigmoid(V'*((U*p_test).*(W*test_data)));
    mAUC_test = mean(computeAUC(pred_test, test_attribute_labels));    
    fprintf(fid, '%0.4f\t %0.4f\t mode:%d\n',mAUC_test_old, mAUC_test, params.mode);

    if(mAUC_val > mAUC_val_old)
        mAUC_val_old = mAUC_val;
        mAUC_test_old = mAUC_test;
        num_cum = 0;
    else
        Opt_W = Opt_W_old; 
        Opt_V = Opt_V_old;
        Opt_U = Opt_U_old;
        Opt_SM_theta = Opt_SM_theta_old;
        params.mode = mod(params.mode, 4) + 1;
        num_cum = num_cum + 1;
        if(num_cum == 4)
            break
        end
    end 
end
