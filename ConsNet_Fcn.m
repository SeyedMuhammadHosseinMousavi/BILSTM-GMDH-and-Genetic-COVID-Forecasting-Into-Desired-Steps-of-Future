function Network2 = ConsNet_Fcn(Network,X)
%%
IW = Network.IW{1,1}; IW_Num = numel(IW);
LW = Network.LW{2,1}; LW_Num = numel(LW);
b1 = Network.b{1,1}; b1_Num = numel(b1);
b2 = Network.b{2,1}; b2_Num = numel(b2);

IWs = X(1:IW_Num); IW = reshape(IWs,size(IW,1),size(IW,2));
LWs = X(IW_Num+1:IW_Num+LW_Num); LW = reshape(LWs,size(LW,1),size(LW,2));
b1s = X(IW_Num+LW_Num+1:IW_Num+LW_Num+b1_Num); b1 = reshape(b1s,size(b1,1),size(b1,2));
b2s = X(IW_Num+LW_Num+b1_Num+1:IW_Num+LW_Num+b1_Num+b2_Num); b2 = reshape(b2s,size(b2,1),size(b2,2));

Network2 = Network;

Network2.IW{1,1} = IW;
Network2.LW{2,1} = LW;
Network2.b{1,1} = b1;
Network2.b{2,1} = b2;

end