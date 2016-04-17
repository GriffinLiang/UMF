% This file is the main program for attribute learning.

% data
load('Awa_Train_decaf.mat');
load('Awa_Test_decaf.mat');
load('AwaPara.mat');
train_data = feaTrain;
test_data = feaTest;
clear feaTrain feaTest 
% TrC:train categories; TeC:test categories; attB: binary attributes; 
% attC: continous attributes; className; nAtt:85; nClass:50; nImClass:1*50
attB(attB == -1) = 0;
NrA = 85; 
NrF = 40;
NrD = size(train_data, 1);
prior = mean(attB(TrC, :) == 1);
% label
train_category_label = processClaLab(TrC, nImClass);
test_category_label = processClaLab(TeC, nImClass);
train_attribute_labels = attB(TrC(train_category_label), :)';
test_attribute_labels = attB(TeC(test_category_label), :)';
val_category_label = train_category_label(mod(1:size(train_data, 2), 10)==0);
train_category_label = train_category_label(mod(1:size(train_data, 2), 10)~=0);
val_data = train_data(:, mod(1:size(train_data, 2), 10)==0);
val_attribute_labels = train_attribute_labels(:, mod(1:size(train_data, 2), 10)==0);
train_attribute_labels = train_attribute_labels(:, mod(1:size(train_data, 2), 10)~=0);
train_data = train_data(:, mod(1:size(train_data, 2), 10)~=0);

fid = 1;
% fid = fopen('Awa_UMF_IS.txt', 'w');


%% Pretrain Stage 1: Attribute Classification
options.Method = 'L-BFGS'; 
options.maxIter = 200;	
options.display = 'on'; 
lambda_pre1 = 1e-5;
w_init = initializeParameters(NrD, NrA);
LogRegOptTheta = w_init;
[LogRegOptTheta, cost] = minFunc( @(p) logRegCost(p, train_data, train_attribute_labels ...
                                        , lambda_pre1), LogRegOptTheta(:), options);        
LogRegOptTheta = reshape(LogRegOptTheta, size(train_data, 1), NrA);
  
pred_val_ac = sigmoid(LogRegOptTheta'*val_data) ;
AUC_val = computeAUC(pred_val_ac, val_attribute_labels);
mAUC_val = mean(AUC_val);
pred_test_ac = sigmoid(LogRegOptTheta'*test_data) ;
AUC_test = computeAUC(pred_test_ac, test_attribute_labels);
mAUC_test = mean(AUC_test);
[zsAcc, zsAcc_norm] = zeroshotDAP(test_category_label, TeC, pred_test_ac, prior, attB);

fprintf(fid, '%d mAUC val: %f, test: %f, zs: %f, zs_n: %f\n', ii, mAUC_val, ...
                                            mAUC_test, zsAcc, zsAcc_norm);

[U_pre1,S_pre1,V_pre1] = svd(LogRegOptTheta);
Opt_pre1_W = U_pre1(:, 1:NrF)*S_pre1(1:NrF, 1:NrF)^(0.5);
Opt_pre1_W = Opt_pre1_W'; 
Opt_pre1_V = S_pre1(1:NrF, 1:NrF)^(0.5)*V_pre1(:, 1:NrF)';

%% Pretrain Stage 2: Category Classification
options.maxIter = 200;	
options.display = 'on'; 
numClasses = 40;
lambda_pre2 = 1e-5;
softmaxModel = softmaxTrain(size(train_data, 1), numClasses, lambda_pre2, ...
                            train_data, train_category_label, options);
SoftmaxOptTheta = softmaxModel.optTheta(:);
softmaxTheta = reshape(SoftmaxOptTheta, numClasses, size(train_data, 1));

M = softmaxTheta*val_data;
[~, p_val] = max(bsxfun(@rdivide, exp(M), sum(exp(M))));
categ_val = mean(p_val == val_category_label');
fprintf(fid, 'Category Acc:%f\n', categ_val);

%% UMF
options.display = 'on'; 
options.maxIter = 200;
lambda = 10.^(-4);
a = 0;
fprintf(fid, 'lambda:%f\ta:%d\n', lambda, a);
Opt_W = Opt_pre1_W; 
Opt_V = Opt_pre1_V;
Opt_U = initializeParameters(NrF, 40);
Opt_SM_theta = SoftmaxOptTheta;
params.NrF = NrF;
params.lambda1 = lambda;
params.lambda2 = lambda;
params.a = a;
params.mode = 1;
label.attribute = train_attribute_labels;
label.category = train_category_label;
mAUC_val_old = 1;
num_cum = 0;
% Only Optimize U
[Opt_U, cost] = minFunc( @(p) UMF_IS_cost(train_data, label, p, Opt_V(:), ...
                      Opt_W(:), Opt_SM_theta(:), params), Opt_U(:), options);

U = reshape(Opt_U, NrF, numClasses);
V = reshape(Opt_V, NrF, NrA);
W = reshape(Opt_W, NrF, NrD);
SM_theta = reshape(Opt_SM_theta, numClasses, NrD);
M = SM_theta*test_data;
M = bsxfun(@minus, M, max(M, [], 1));
p_test = bsxfun(@rdivide, exp(M), sum(exp(M)));
pred_test = sigmoid(V'*((U*p_test).*(W*test_data)));
[zsAcc, zsAcc_norm] = zeroshotDAP(test_category_label, TeC, pred_test, prior, attB);
fprintf(fid, 'Test mAUC_val:%0.4f Accuracy:%0.4f %0.4f\n', mAUC_val, zsAcc, zsAcc_norm);    