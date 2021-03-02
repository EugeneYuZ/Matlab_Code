clear; close all; clc;
load('data.mat');
data = [m;n];

rng(2);
shuffle_data = data(randperm(size(data, 1)), :);

x0 = shuffle_data(:, 1:60);
data_features = (x0-min(x0(:))) ./ (max(x0(:))-min(x0(:)));
labels = shuffle_data(:, 61);

k = 5;
cvFolds = crossvalind('Kfold', labels, k); 

for i = 1:k                                
    testIdx = (cvFolds == i);  % get indices of test instances
    trainIdx = ~testIdx;       % get indices training instances
    mdlSVM  = fitcsvm(data_features(trainIdx,:),labels(trainIdx),'Standardize',true);
    mdlSVM = fitPosterior(mdlSVM);
    [lab,score_svm] = resubPredict(mdlSVM);
    
    % the last 1 is the positve class label, here we use double 1 as the last argument. 
    [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(labels(trainIdx,:), score_svm(:,2),1);
    plot(Xsvm,Ysvm)
    xlabel('False positive rate'); ylabel('True positive rate');
    title('ROC Curves for SVM')
end



