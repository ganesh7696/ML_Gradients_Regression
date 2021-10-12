
function [X_train, X_test, Y_train, Y_test,standarizedX] = loadData
%Loading Data and Standarizing it
%   Detailed explanation goes here
clc;
clear;
data = importdata("insurance.csv");
Y = data.data;
data = data.textdata;
data(1,:)=[];
data(:,6) =[];
data(:,6) =[];
[uniquesex, ~, sex]=unique(data(:,2));
[uniqueSmk, ~, Smk] = unique(data(:,5));
data(:,2) = num2cell(sex);
data(:,5) = num2cell(Smk);
data(:,1)=num2cell(str2double(data(:,1)));
data(:,3)=num2cell(str2double(data(:,3)));
data(:,4)=num2cell(str2double(data(:,4)));
X= cell2mat(data);
m = length(X);
muX = mean(X);
stdX = std(X);
repstd = repmat(stdX, m, 1);
repmu = repmat(muX,m,1);
standarizedX = (X-repmu)./repstd;
standarizedX(:,6) = ones(m,1);
cv=cvpartition(m,'HoldOut',0.5);
idx = cv.test;
X_train = standarizedX(~idx,:);
X_test = standarizedX(idx,:);
Y_train = Y(~idx,:);
Y_test = Y(idx,:);
end
