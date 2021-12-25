clc;
clear;
close all;

%% Input
x=load('hopkinsirandeath.txt');
Delays = [10 20 30 40 50];
[Inputs, Targets] = CreateTimeSeriesData(x,Delays);
nData = size(Inputs,2);
Perm = randperm(nData);
% Train 
pTrain = 0.8;
nTrainData = round(pTrain*nData);
TrainInd = Perm(1:nTrainData);
TrainInputs = Inputs(:,TrainInd);
TrainTargets = Targets(:,TrainInd);
% Test 
pTest = 1 - pTrain;
nTestData = nData - nTrainData;
TestInd = Perm(nTrainData+1:end);
TestInputs = Inputs(:,TestInd);
TestTargets = Targets(:,TestInd);

% Create GMDH Network
params.MaxLayerNeurons = 30;   % Maximum Number of Neurons in a Layer
params.MaxLayers = 9;          % Maximum Number of Layers
params.alpha = 0;              % Selection Pressure
params.pTrain = 0.2;           % Train Ratio

% Tran GMDH
gmdh = GMDH(params, TrainInputs, TrainTargets);

%  GMDH Model on Train and Test (Validation)
Outputs = ApplyGMDH(gmdh, Inputs);
TrainOutputs = Outputs(:,TrainInd);
TestOutputs = Outputs(:,TestInd);


%% Predict Plots
% figure;
% PlotResults(TrainTargets, TrainOutputs, 'Train Data');
% figure;
% PlotResults(TestTargets, TestOutputs, 'Test Data');
figure;
set(gcf, 'Position',  [50, 100, 1200, 450])
[MSE RMSE ErrorMean ErrorStd Errors]=PlotResults(Targets, Outputs, 'All Data');
figure;
plotregression(TrainTargets, TrainOutputs, 'Train Data', ...
               TestTargets, TestOutputs, 'TestData', ...
               Targets, Outputs, 'GMDH All Data');

% Forecast Plot
fore=180;
sizee=size(x);
sizee=sizee(1,2);
forecasted=Outputs(1,end-fore:end);
forecasted=forecasted+x(end)/2.5;
ylbl=sizee+fore;
t = linspace(sizee,ylbl,length(forecasted));

% Compare the simulated output with measured data to ensure it is a good fit.
nstep=32;
sys = nlarx(Outputs',64);
figure;
set(gcf, 'Position',  [50, 200, 1300, 400])
compare(Outputs',sys,nstep);title('Covid Iran Death');
grid on;
%
figure;
regres=compare(Outputs',sys,nstep);title('Covid Iran Death');
b1 = regres\Outputs';
yCalc1 = b1*regres;
scatter(regres,Outputs','MarkerEdgeColor',[0 .5 .5],...
              'MarkerFaceColor',[0 .8 .8],...
              'LineWidth',1);
hold on
plot(regres,yCalc1,':',...
    'LineWidth',2,...
    'MarkerSize',5,...
    'Color',[0.6350 0.0780 0.1840]);
title(['Regression :   ' num2str(b1) ])
grid on

%
figure;
set(gcf, 'Position',  [20, 20, 1000, 250])
plot(x,'--',...
    'LineWidth',1,...
    'MarkerSize',5,...
    'Color',[0,0,0.7]);
hold on;
plot(t,forecasted,'-.',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'Color',[0.9,0.5,0]);
title('Johns Hopkins Data for Iran COVID Deaths - Orange is Forcasted')
xlabel('Days - From Jan 2020 Till Dec 2021','FontSize',12,...
       'FontWeight','bold','Color','b');
ylabel('Number of People','FontSize',12,...
       'FontWeight','bold','Color','b');
   datetick('x','mmm');
legend({'Measured','GMDH Forecasted'});
%% Metrics
% Explained Variance (EV)
va=var(Outputs-Targets);vc=var(Targets);
vd=abs(va/vc);vb = vd / 10^floor(log10(vd));
EV=vb*0.1
% MSE
MSE 
% RMSE
RMSE
% Mean Error
ErrorMean
% STD Error
ErrorStd 
% Mean Absolute Error (MAE)
MAE = mae(Targets,Outputs)
% Root Mean Squared Log Error (RMSLE) 
RMSLE = sum((log(Targets)-log(Outputs)).^2)*0.1
