function SelectedParentsIndex = MySelection_Fcn(Cost,NumOfSelection,Mode)
PopSize = numel(Cost);

switch Mode
    case 1 % Random Selection
        R = randperm(PopSize);
        SelectedParentsIndex = R(1:NumOfSelection);
        
    case 2 % Tournment Selection
        for ii = 1:NumOfSelection
            R = randperm(PopSize);
            Sel = R(1:2);
            SelCost = Cost(Sel);
            Inx = find(SelCost == min(SelCost)); Inx = Inx(1);
            SelectedParentsIndex(ii) = Sel(Inx);
        end
        
    case 3 % ICA Method
        for ii = 1:NumOfSelection
            CostN = 1.1*max(Cost) - Cost;
            P = CostN / sum(CostN);
            R = rand(PopSize,1);
            D = P - R;
            Inx = find(D == max(D)); Inx = Inx(1);
            SelectedParentsIndex(ii) = Inx;
        end
end
