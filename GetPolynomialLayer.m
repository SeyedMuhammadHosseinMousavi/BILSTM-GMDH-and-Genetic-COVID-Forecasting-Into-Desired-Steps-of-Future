function L = GetPolynomialLayer(X1, Y1, X2, Y2)

    n = size(X1,1);
    
    N = n*(n-1)/2;
    
    template = FitPolynomial(rand(2,3),rand(1,3),rand(2,3),rand(1,3),[]);
    
    L = repmat(template, N, 1);
    
    k = 0;
    for i=1:n-1
        for j=i+1:n
            k = k+1;
            L(k) = FitPolynomial(X1([i j],:), Y1, X2([i j],:), Y2, [i j]);
        end
    end
    
    [~, SortOrder] = sort([L.RMSE2]);

    L = L(SortOrder);
    
end