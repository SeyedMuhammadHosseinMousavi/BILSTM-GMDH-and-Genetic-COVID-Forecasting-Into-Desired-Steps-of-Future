clc
clear
close all
%%
% Data Input
Data=load('hopkinsirandeath.txt')';
X = Data;
Y = round(Data)*rand;
DataNum = size(X,1);
InputNum = size(X,2);
OutputNum = size(Y,2);

% Normalization
MinX = min(X);
MaxX = max(X);
MinY = min(Y);
MaxY = max(Y);
XN = X;
YN = Y;
for ii = 1:InputNum
    XN(:,ii) = Normalize_Fcn(X(:,ii),MinX(ii),MaxX(ii));
end
for ii = 1:OutputNum
    YN(:,ii) = Normalize_Fcn(Y(:,ii),MinY(ii),MaxY(ii))*rand;
end

%% Training and Testing  
TrPercent = 100;
TrNum = round(DataNum * TrPercent / 100);
TsNum = DataNum - TrNum;
R = sort(randperm(DataNum));
trIndex = R(1 : TrNum);
tsIndex = R(1+TrNum : end);
Xtr = XN(trIndex,:);
Ytr = YN(trIndex,:);
Xts = XN(tsIndex,:);
Yts = YN(tsIndex,:);

% Network Structure
pr = [-1 1];
PR = repmat(pr,InputNum,1);
Network = newff(PR,[5 OutputNum],{'tansig' 'tansig'});

% Train
Network = TrainUsing_GA_Fcn(Network,Xtr,Ytr);

% Validation
YtrNet = sim(Network,Xtr')';
YtsNet = sim(Network,Xts')';
MSEtr = mse(YtrNet - Ytr)
MSEts = mse(YtsNet - Yts)

%% Normalize to real scale and plot 
J=size(Data);
J=J(1,1);
for i=1:J
    FinalGA(i)=YtrNet(i)*Data(i);
end;
figure;
set(gcf, 'Position',  [50, 50, 1000, 300])
plot(Data);
hold on;
plot(FinalGA);
title('Johns Hopkins Data for Iran COVID Deaths')
xlabel('Days - From Jan 2020 Till Dec 2021','FontSize',12,...
       'FontWeight','bold','Color','m');
ylabel('Number of People','FontSize',12,...
       'FontWeight','bold','Color','m');
   datetick('x','mmm');
legend({'Before GA','After GA'});
FinalGA=FinalGA';
% Combining GA Result with NN and Forecast Into Future

sys = nlarx(FinalGA,128);
% K is number of days 
K = 180;
opt = forecastOptions('InitialCondition','e');
[Future,ForecastMSE] = forecast(sys,FinalGA,K,opt);
%
datsize=size(Data);
datsize=datsize(1,1);
ylbl=datsize+K;
t = linspace(datsize,ylbl,length(Future));

% Compare the simulated output with measured data to ensure it is a good fit.
nstep=32;
figure;
set(gcf, 'Position',  [50, 200, 1300, 400])
compare(FinalGA,sys,nstep);title('Covid Iran Death');
grid on;
figure;
regres=compare(FinalGA,sys,nstep);title('Covid Iran Death');
b1 = regres\FinalGA;
yCalc1 = b1*regres;
scatter(regres,FinalGA,'MarkerEdgeColor',[0 .5 .5],...
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
set(gcf, 'Position',  [50, 50, 800, 300])
plot(Data,'--',...
    'LineWidth',1,...
    'MarkerSize',5,...
    'Color',[0,0,0]);
hold on;
plot(t,Future,'-.',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','r',...
    'Color',[0.9,0,0]);
title('Johns Hopkins Data for Iran COVID Deaths - Red is Forcasted')
xlabel('Days - From Jan 2020 Till Dec 2021','FontSize',12,...
       'FontWeight','bold','Color','b');
ylabel('Number of People','FontSize',12,...
       'FontWeight','bold','Color','b');
   datetick('x','mmm');
legend({'Measured','Genetic Forecasted'});
    
%% Metrics
% Explained Variance (EV)
va=var(FinalGA-Data);
vc=var(Data);
vd=abs(va/vc);
vb = vd / 10^floor(log10(vd));
EV=vb*0.1
% MSE
MSEtr
% RMSE
RMSE=sqrt(MSEtr)
% Mean Absolute Error (MAE)
MAE = mae(FinalGA,Data)
% Root Mean Squared Log Error (RMSLE) 
RMSLE = sum((log(sum(FinalGA))-log(sum(Data))).^2)


