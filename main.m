
[X_train, X_test, Y_train, Y_test,standarizedX] = loadData;
batch_Gradient(standarizedX,X_train,X_test,Y_train,Y_test,'');
batch_Gradient(standarizedX,X_train,X_test,Y_train,Y_test,'l1');
batch_Gradient(standarizedX,X_train,X_test,Y_train,Y_test,'l2');
stochastic_Gradient(standarizedX,X_train,X_test,Y_train,Y_test,'')
mini_Batch_Gradient(standarizedX,X_train,X_test,Y_train,Y_test,'');
