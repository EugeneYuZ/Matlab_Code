% plot roc
function  auc = plot_roc( predict, ground_truth )  
x = 1.0;  
y = 1.0;  
pos_num = sum(ground_truth==1);  
neg_num = sum(ground_truth==0);  
 
x_step = 1.0/neg_num;  
y_step = 1.0/pos_num;   
[predict,index] = sort(predict);  
ground_truth = ground_truth(index);  
for i=1:length(ground_truth)  
    if ground_truth(i) == 1  
        y = y - y_step;  
    else  
        x = x - x_step;  
    end  
    X(i)=x;  
    Y(i)=y;  
end  
     
plot(X,Y,'-ro','LineWidth',1,'MarkerSize',1);  
xlabel('Specificity(False positive rate)');  
ylabel('Sensitivity(True positive rate)');  
title('ROC Curve');   
auc = -trapz(X,Y);            
end  
