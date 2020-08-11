
% ----------------------------------------------------------------------- %
% 
% Plot curves 
% Created by T. Zhou
% E-mail: taozhou.ai@gmail.com
% 
% ----------------------------------------------------------------------- %
clc;clear;close all;


disp('***************** ---------------------------------------------------')
disp('Note that the mean evaluation results of each dataset are saved in <per_metric.mat> ')
disp('***************** ---------------------------------------------------')

plotDrawStyle24={   struct('color',[1,0,0],'lineStyle','-'),...
    struct('color',[0,1,0],'lineStyle','-'),...
    struct('color',[0,0,1],'lineStyle','-'),...
    struct('color',[0,0,0],'lineStyle','-'),...%    struct('color',[1,1,0],'lineStyle','-'),...%yellow
    struct('color',[1,0,1],'lineStyle','-'),...%pink
    struct('color',[1,1,0],'lineStyle','-'),...%
    struct('color',[0,1,1],'lineStyle','-'),...
    struct('color',[0.5,0.5,0.5],'lineStyle','-'),...%gray-25%
    struct('color',[136,0,21]/255,'lineStyle','-'),...%dark red
    struct('color',[255,127,39]/255,'lineStyle','-'),...%orange
    struct('color',[0,162,232]/255,'lineStyle','-'),...%Turquoise
    struct('color',[163,73,164]/255,'lineStyle','-'),...%purple    %%%%%%%%%%%%%%%%%%%%
    struct('color',[1,0,0],'lineStyle','-.'),...
    struct('color',[0,1,0],'lineStyle','-.'),...
    struct('color',[0,0,1],'lineStyle','-.'),...
    struct('color',[0,0,0],'lineStyle','-.'),...%    struct('color',[1,1,0],'lineStyle',':'),...%yellow
    struct('color',[1,0,1],'lineStyle','-.'),...%pink
    struct('color',[1,1,0],'lineStyle','-.'),...%
    struct('color',[0,1,1],'lineStyle','-.'),...
    struct('color',[0.5,0.5,0.5],'lineStyle','-.'),...%gray-25%
    struct('color',[136,0,21]/255,'lineStyle','-.'),...%dark red
    struct('color',[255,127,39]/255,'lineStyle','-.'),...%orange
    struct('color',[0,162,232]/255,'lineStyle','-.'),...%Turquoise
    struct('color',[163,73,164]/255,'lineStyle','-.'),...%purple
    };

% ----------------------------------------------------------------------- %
metric_path = 'results/Sal_Det_Results_24_Models/';
methods     = {'LHM','ACSD','DESM','GP','LBE','DCMC','SE','CDCP','CDB','DF','PCF','CTMF','CPFP','TANet','AFNet','MMCI','DMRA','D3Net','SSF','A2dele','S2MA','ICNet','JL-DCF','UCNet'};
data_title  = {'STERE', 'GIT', 'DES', 'NLPR', 'LFSD', 'NJUD', 'SSD', 'SIP'};


% -------------------------------------- %
zeros_tem = zeros(1,256);


for ind_data = 1:length(data_title)
    
    plotMetrics_Recall = [];
    plotMetrics_Pre    = [];
    Fmeasure_Curve     = [];
    
    % ------------------------------------------------------------------- % 
        
    for ind_model = 1:length(methods)
        cur_model = methods{ind_model};
        disp(['process:' cur_model]);
        
        
        % --------------------------------------------------------------- %
        cur_data = data_title{ind_data};
        cur_metric_path = [metric_path,cur_model,'/'];
        
        if exist([cur_metric_path,cur_data,'_metrics.mat']) == 0 
            plotMetrics_Recall = [plotMetrics_Recall; zeros_tem];
            plotMetrics_Pre    = [plotMetrics_Pre; zeros_tem];
            Fmeasure_Curve     = [Fmeasure_Curve; zeros_tem];
            
        else
            load([cur_metric_path,cur_data,'_metrics.mat']);
        
            plotMetrics_Recall = [plotMetrics_Recall; per_metric.Recall];
            plotMetrics_Pre    = [plotMetrics_Pre; per_metric.Pre];
            Fmeasure_Curve     = [Fmeasure_Curve;per_metric.Fmeasure_Curve];
        end
       
    end
    
    % ------------------------------------------------------------------- % 
    % **************** plot the PR curves ******************************* %
    % ------------------------------------------------------------------- % 
    figure(ind_data*2-1);
    hold on;
    grid on;

    axis([0 1 0 1]);
    set(gca,'FontSize',16);

    xlabel('Recall','fontsize',24); ylabel('Precision','fontsize',24);
    width = 4;

    for i = 1:size(plotMetrics_Pre,1)
        if plotMetrics_Recall(i,1) == 0
            continue;
        end
        h1(i)=plot(plotMetrics_Recall(i,:), plotMetrics_Pre(i,:), 'color',plotDrawStyle24{25-i}.color, 'lineStyle', plotDrawStyle24{25-i}.lineStyle,'lineWidth', width);
           
    end
    title(cur_data)
    set(gca,'linewidth',2);


    % ------------------------------------------------------------------- % 
    % **************** plot the Fmeasure curves ************************* %
    % ------------------------------------------------------------------- % 
    figure(ind_data*2);
    hold on;
    grid on;
    axis([0 255 0 1]);
    set(gca,'FontSize',16);

    xlabel('Threshold','fontsize',24);
    ylabel('F-measure','fontsize',24);
    x = [255:-1:0]';

    width = 4;

    for i = 1:size(Fmeasure_Curve,1)
        if Fmeasure_Curve(i,1) == 0
            continue;
        end
        
        h2(i)=plot(x,  Fmeasure_Curve(i,:), 'color',plotDrawStyle24{25-i}.color, 'lineStyle', plotDrawStyle24{25-i}.lineStyle,'lineWidth', width);
    
    end
    title(cur_data)
    set(gca,'linewidth',2);
    
    % -------------------------------------- legend 
%     lgd1 = legend(h2(1:12),{'LHM','ACSD','DESM','GP','LBE','DCMC','SE','CDCP','CDB','DF','PCF','CTMF'});
%     ah   = axes('position',get(gca,'position'),'visible','off');
%     lgd2 = legend(ah,h2(13:24),{'CPFP','TANet','AFNet','MMCI','DMRA','D3Net','SSF','A2dele','S2MA','ICNet','JL-DCF','UCNet'},'location','west');


end

