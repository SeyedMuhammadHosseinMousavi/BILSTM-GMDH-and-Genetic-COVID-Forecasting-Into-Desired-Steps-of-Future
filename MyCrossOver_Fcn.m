function [Off1 , Off2] = MyCrossOver_Fcn(Parent1,Parent2)

Beta1 = rand(1,numel(Parent1));
Off1 = Beta1 .* Parent1 + (1 - Beta1) .* Parent2;

Beta2 = rand(1,numel(Parent1));
Off2 = Beta2 .* Parent1 + (1 - Beta2) .* Parent2;