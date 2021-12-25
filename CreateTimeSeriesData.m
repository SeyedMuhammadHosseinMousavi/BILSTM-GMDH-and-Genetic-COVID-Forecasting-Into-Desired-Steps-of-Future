function [X, Y] = CreateTimeSeriesData(x, Delays)

    T = size(x,2);
    
    MaxDelay = max(Delays);
    
    Range = MaxDelay+1:T;
    
    X= [];
    for d = Delays
        X=[X; x(:,Range-d)];
    end
    
    Y = x(:,Range);

end