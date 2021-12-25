clear;
data=load('hopkinsirandeath.txt');
data=data(:,153:end);
% Devide train and test
numTimeStepsTrain = floor(0.65*numel(data));
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);
% At prediction time, you must standardize the test data using the 
% same parameters as the training data.
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;
% Prepare Predictors and Responses
% To forecast the values of future time steps of a sequence, specify the responses
% to be the training sequences with values shifted by one time step.
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
% Define LSTM Network Architecture
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 100;
layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
% Specify the training options. 
options = trainingOptions('rmsprop', ...
    'MaxEpochs',100, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.06, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',128, ...
    'LearnRateDropFactor',0.02, ...
    'Verbose',0, ...
    'Plots','training-progress');
% Train LSTM Network
net = trainNetwork(XTrain,YTrain,layers,options);

%% Forecast Future Time Steps
% Standardize the test data using the same parameters as the training data.
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);
[net,YPred]  = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end
% Unstandardize the predictions using the parameters calculated earlier.
YPred = sig*YPred + mu;
% The training progress plot reports the root-mean-square error (RMSE) calculated 
% from the standardized data. Calculate the RMSE from the unstandardized predictions.
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))
% Plot the training time series with the forecasted values.
figure;
plot(dataTrain(1:end-1));
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-');
hold off
xlabel("Month");
ylabel("Cases");
title("BI-LSTM Forecast");
legend(["Observed" "BI-LSTM Forecast"]);
% Compare the forecasted values with the test data.
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "BI-LSTM Forecast"])
ylabel("Cases")
title("BI-LSTM Forecast")
subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)
%
%% Metrics
% Explained Variance (EV)
va=var(YPred-YTest);
vc=var(YTest);
vd=abs(va/vc);
vb = vd / 10^floor(log10(vd));
EV=vb*0.1
% MSE
MSE=mean((YPred-YTest).^2)
% RMSE
rmse
% Mean Absolute Error (MAE)
MAE = mae(YPred,YTest)
% Root Mean Squared Log Error (RMSLE) 
RMSLE = sum((log(sum(YPred))-log(sum(YTest))).^2)