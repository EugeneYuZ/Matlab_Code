
clear; close all; clc;
load('data.mat');

%shuffle
rng(2);
shuffle_data = data(randperm(size(data, 1)), :);
indices = crossvalind('Kfold', size(shuffle_data,1), 5); 
cp = classperf(size(shuffle_data));

x0 = shuffle_data(:, 1:60);
x = (x0-min(x0(:))) ./ (max(x0(:))-min(x0(:)));
y = shuffle_data(:, 61);

% normal splitting 
% k=0.7*size(data,1);
% X1=x(1:k,:);
% Y1=y(1:k,:);
% X2=x(k+1:end,:);
% Y2=y(k+1:end,:);
% 
% [m, n] =size(X1);
% indices=crossvalind('Kfold',data(1:M,N),5);

[m, n] = size(x);
m = 4/5*m;
theta1 = zeros(n+1,1);
delta=1e-1;  
num = 1000; 
L=[];
acc_array=[];
TP = [];
TN = [];
FP= [];
FN = [];


for i = 1:5
   test = (indices == i); 
   train = ~test;
   
   % train
   xx = [x(train,:) ones(m,1)];
   yy = y(train,:);
   while(num)
        h = sigmoid(xx*theta1); 
        loss = -(1/m)*sum(log(sigmoid((2*yy - 1).*xx*theta1))); 
        for i = 1 : size(theta1, 1)
            dt(i) = (1/m)*sum(sigmoid((1-2*yy).*xx*theta1).*(1-2*yy).*xx(:, i));
        end
        L=[L,loss];
        theta2=theta1 - delta*dt';
        theta1=theta2;
        num = num - 1;
        if loss <0.02
            break;
        end
   end
   
   % test
   acc = 0;
   true_positive = 0;
   true_negative = 0;
   false_positive = 0;
   false_negative = 0;
   
   [m_test, n_test] = size(x(test,:));
   x_test = [x(test,:) ones(m_test,1)];
   y_test = y(test,:);
   
   for i=1:m_test
        xx1=x_test(i,:);
        yy=y_test(i);
        finil=1/(1+exp(-xx1*theta2 ));
        
        %choose 0.5 as threshold
        if finil>0.5 && yy==1
            acc=acc+1;
            true_positive = true_positive + 1;
        end
        if finil<=0.5 && yy==0
            acc=acc+1;
            true_negative = true_negative + 1;
        end
        if finil<=0.5 && yy==1
            false_negative = false_negative + 1;
        end
        if finil>0.5 && yy==0
            false_positive = false_positive + 1;
        end
   end
   
   acc_array =[acc_array, acc/m_test];
   TP = [TP,true_positive/m_test];
   TN = [TN,true_negative/m_test];
   FP= [FP,false_positive/m_test];
   FN = [FN,false_negative/m_test];
end
   
 acc_array_f = mean(acc_array);
 TP_f = mean(TP);
 TN_f = mean(TN);
 FP_f = mean(FP);
 FN_f = mean(FN);
 
Sensitivity = TP_f/(TP_f+FN_f);
Specificity = TN_f/(TN_f+FP_f);
 
 predict = 1./(1+exp(-x_test*theta2 ));
   
 result=plot_roc(predict,y_test);  
 disp(result);  

 
 











    


