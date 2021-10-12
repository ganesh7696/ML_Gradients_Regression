
function  batch_Gradient(standarizedX,X_train, X_test, Y_train, Y_test,regularization)
%Batch Gradient Implementation
%   Detailed explanation goes here
m = length(X_train);     %. Training size
err_threshold = 0;     % Threshold to run number of epochs to reach minimum found
w = [10 9 8 7 6 5];      % Random Initialization of w
alpha = 0.1; %penalty coefficient to minimize overfitting
count =0;
% For next step assigned  err_threshold to be 2264782890.564bound on running 20epochs to verify error minimum
while(err_threshold == 0 || err_threshold >= 2264782890.564 && count<=20)
    lambda = 0.1;   % Learning rate to find the global minima
    Y_hat = zeros(m,1);
    for j = 1: m
        Y_hat(j) = dot(w,X_train(j,:)); % Finding y_hat using dot product of weight and training data instance
    end
    training_MSE = (Y_train - Y_hat).^2; % MSE of training data
    if regularization == "l1"
        gradient_MSE = 2/m * (X_train)'*((X_train * w') - Y_train) ;  % Finding gradient(derivative) of MSE using L1 regularization
        for i = 1:length(gradient_MSE)
            gradient_MSE(i) =  gradient_MSE(i) + 2*alpha*sign(w(i));
        end
        w = w - (lambda * gradient_MSE');  %Updating weight vector based on gradientMSE after  data points processed
    elseif regularization == "l2"
        gradient_MSE = 2/m * (X_train)'*((X_train * w') - Y_train) ;% Finding gradient(derivative) of MSE  using L2 regularization
        for i = 1:length(gradient_MSE)
            gradient_MSE(i) =  gradient_MSE(i) + 2*alpha*w(i);
        end
        w = w - (lambda * gradient_MSE'); %Updating weight vector based on gradientMSE after  data points processed
        min_idx =w == min(w);
        w(min_idx) =0;
        regularization='';
    else
        gradient_MSE = 2/m * (X_train)'*((X_train * w') - Y_train); % Finding gradient(derivative) of MSE
        w = w - (lambda * gradient_MSE');   %Updating weight vector based on gradientMSE after data points processed
    end
    err_threshold=norm(training_MSE); %Norm of Mean Squared Error
    disp("\n MSE AFTER EACH STEP USING GRADIENT DESCENT");
    disp(num2str(err_threshold)); % Displaying error after each iteration
    count=count+1;
end
disp("**************");
disp("Printing final weight value");
disp(w);
for n = 1:length(X_test)
    Y_pred = dot(w,X_test(n,:));
end
disp("\n TEST DATA MSE");
testing_MSE = (Y_test - Y_pred).^2;
disp(num2str(norm(testing_MSE)));
X = standarizedX;
%%%plotting Liner Regression as a function of Attributes
for l = 1: length(standarizedX)
     x1 = X(l,1);
     A(l) =dot(w, [x1 1 1 1 1 1]);%linear regression as a function of age
     x2=X(l,2);
     B(l)=dot(w, [1 x2 1 1 1 1]);% linear regression as a function of sex
     x3=X(l,3);
     C(l)=dot(w, [1 1 x3 1 1 1]); % linear regression as a function of bmi
     x4=X(l,4);
     D(l)=dot(w, [1 1 1 x4 1 1]); % linear regression as a function of children
     x5= X(l,5);
     E(l)=dot(w, [1 1 1 1 x5 1]); % linear regression as a function of smoker
end
figure()
title("Linear Regression")
subplot(3,3,1)
disp(length(A));
plot(X(:,1),A)
xlabel("Age")
ylabel("Linear Regression function of Age")
subplot(3,3,2)
plot(X(:,2),B)
xlabel("Sex")
ylabel("Linear Regression function of Sex")
subplot(3,3,3)
plot(X(:,3),C)
xlabel("BMI")
ylabel("Linear Regression function of BMI")
subplot(3,3,7)
plot(X(:,4),D)
xlabel("Number of Children")
ylabel("Linear Regression function of Number of Children")
subplot(3,3,8)
plot(X(:,5),E)
xlabel("Smoker")
ylabel("Linear Regression function of Smoker")
end
