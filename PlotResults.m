function [MSE RMSE ErrorMean ErrorStd Errors]=PlotResults(Targets, Outputs, Title)

    Errors = Targets - Outputs;
    MSE = mean(Errors.^2);
    RMSE = sqrt(MSE);
    ErrorMean = mean(Errors);
    ErrorStd = std(Errors);
    
    subplot(2,2,[1 2]);
    plot(Targets,'-.',...
    'LineWidth',1,...
    'MarkerSize',10,...
    'Color',[0.9,0.0,0.0]);
    hold on;
    plot(Outputs,'--',...
    'LineWidth',1,...
    'MarkerSize',10,...
    'Color',[0.1,0.5,0]);
    legend('Targets','Outputs');
    ylabel('Targets and Outputs');
    grid on;
    title(Title);
    
    subplot(2,2,3);
    plot(Errors,'-.',...
    'LineWidth',1.5,...
    'MarkerSize',5,...
    'Color',[0.0,0.2,0.2]);
    title(['MSE = ' num2str(MSE) ', RMSE = ' num2str(RMSE)]);
    ylabel('Errors');
    grid on;
    
    subplot(2,2,4);
    h=histfit(Errors, 80);
    h(1).FaceColor = [.9 .5 1];
    h(2).Color = [.2 .9 .2];
    title(['Error Mean = ' num2str(ErrorMean) ', Error StD = ' num2str(ErrorStd)]);

end