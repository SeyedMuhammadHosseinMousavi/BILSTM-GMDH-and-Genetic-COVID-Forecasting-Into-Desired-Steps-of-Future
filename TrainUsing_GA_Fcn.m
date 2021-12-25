function [Network2] = TrainUsing_GA_Fcn(Network,Xtr,Ytr)


%% Problem Statement
IW = Network.IW{1,1}; IW_Num = numel(IW);
LW = Network.LW{2,1}; LW_Num = numel(LW);
b1 = Network.b{1,1}; b1_Num = numel(b1);
b2 = Network.b{2,1}; b2_Num = numel(b2);

TotalNum = IW_Num + LW_Num + b1_Num + b2_Num;

NPar = TotalNum;

VarLow = -1;
VarHigh = 10;
FunName = 'Cost_ANN_EA';

%% Algorithm Parameters
SelectionMode = 2; % 1 for Random, 2 for Tournment, 3 for ....
PopSize = 20;
MaxGenerations = 20;

RecomPercent = 15/100;
CrossPercent = 50/100;
MutatPercent = 1 - RecomPercent - CrossPercent;

RecomNum = round(PopSize*RecomPercent);
CrossNum = round(PopSize*CrossPercent);
if mod(CrossNum,2)~=0
    CrossNum = CrossNum - 1;
end

MutatNum = PopSize - RecomNum - CrossNum;

%% Initial Population
Pop = rand(PopSize,NPar) * (VarHigh - VarLow) + VarLow;

Cost = feval(FunName,Pop,Xtr,Ytr,Network);
[Cost Inx] = sort(Cost);
Pop = Pop(Inx,:);

%% Main Loop
MinCostMat = [];
MeanCostMat = [];

for Iter = 1:MaxGenerations
    %% Recombination
    RecomPop = Pop(1:RecomNum,:);
    
    %% CrossOver
        %% Parent Selection
        SelectedParentsIndex = MySelection_Fcn(Cost,CrossNum,SelectionMode);
    
        %% Cross Over
        CrossPop = [];
        for ii = 1:2:CrossNum
            Par1Inx = SelectedParentsIndex(ii);
            Par2Inx = SelectedParentsIndex(ii+1);

            Parent1 = Pop(Par1Inx,:);
            Parent2 = Pop(Par2Inx,:);
            

            [Off1 , Off2] = MyCrossOver_Fcn(Parent1,Parent2);
            
            CrossPop = [CrossPop ; Off1 ; Off2];
        end
    %% Mutation
    MutatPop = rand(MutatNum,NPar)*(VarHigh - VarLow) + VarLow;
    
    %% New Population
    Pop = [RecomPop ; CrossPop ; MutatPop];
    Cost = feval(FunName,Pop,Xtr,Ytr,Network);
    [Cost Inx] = sort(Cost);
    Pop = Pop(Inx,:);
   
    %% Display
    MinCostMat = [MinCostMat ; min(Cost)];
    [Iter MinCostMat(end)]
    MeanCostMat = [MeanCostMat ; mean(Cost)];
    subplot(2,1,1)
    plot(MinCostMat,'c','linewidth',1.5);
        title('Genetic Algorithm Training');
    xlim([1 MaxGenerations])
%     hold on
%     plot(MeanCostMat,':b','linewidth',2)
%     hold off
    
    subplot(2,1,2)
    plot(Pop(:,1),Pop(:,2),'*')
        title('Population Scattering');
    axis([VarLow VarHigh VarLow VarHigh])
    pause(0.05)
    
end
%% Final Result Demonstration
BestSolution = Pop(1,:);
BestCost = Cost(1);
Network2 = ConsNet_Fcn(Network,BestSolution);
