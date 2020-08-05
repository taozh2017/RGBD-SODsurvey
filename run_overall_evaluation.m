
% ----------------------------------------------------------------------- %
% 
% Overall evaluation using S_aplha and MAE metrics
% Created by T. Zhou
% E-mail: taozhou.ai@gmail.com
% 
% ----------------------------------------------------------------------- %
clc;clear;close all;

disp('***************** ---------------------------------------------------')
disp('pls download *Sal_Det_Results_24_Models* and put it into *results*')
disp('***************** ---------------------------------------------------')

% ------------------------------ load pre-computed results 
sal_results_path  = 'results/Sal_Det_Results_24_Models/';

% ------------------------------ evaluation_methods and datasets
evaluation_methods     = {'LHM','ACSD','DESM','GP','LBE','DCMC','SE','CDCP','CDB','DF','PCF','CTMF','CPFP','TANet','AFNet','MMCI','DMRA','D3Net','SSF','A2dele','S2MA','ICNet','JL-DCF','UCNet'};
evaluation_methods_sub = {'D3Net','SSF','S2MA','ICNet','JL-DCF','UCNet'};
datasets               = {'STERE', 'NLPR', 'LFSD','DES','SIP'};


% -------------------------------------------------- %
% compute mean values of s_alpha and MAE 
% -------------------------------------------------- %

MAE_overall    = [];
Salpha_overall = [];

for ind1 = 1:length(evaluation_methods)
    
    MAE_data_set    = [];
    Salpha_data_set = [];
    for ind2 =1:length(datasets)
        
        % ------------------------------
        cur_model = evaluation_methods{ind1};
        cur_data  = datasets{ind2};
        
        disp(['process:' cur_model, ' ----> on dataset:' cur_data]);
        
        cur_model_sal_path = [sal_results_path,cur_model,'/',cur_data,'/'];
        dirs = dir([cur_model_sal_path,'*.mat']);
        
        % ------------------------------ obatin results on the current dataset
        MAE_set    = [];
        Salpha_set = [];
        for ind3 = 1:length(dirs)
            
            cur_image_results_path = [cur_model_sal_path,dirs(ind3).name];
            load(cur_image_results_path);
            
            MAE_set    = [MAE_set;eva_metrics_results.MAE];
            Salpha_set = [Salpha_set;eva_metrics_results.Smeasure];
        end
        
        MAE_data_set    = [MAE_data_set;(MAE_set)];
        Salpha_data_set = [Salpha_data_set;(Salpha_set)];
         
        
    end
    
    % ------------------------------ %
    MAE_overall    = [MAE_overall,MAE_data_set];
    Salpha_overall = [Salpha_overall,Salpha_data_set];
    
end


% -------------------------------------------------- %
% show overall evaluation 
% -------------------------------------------------- %

y = mean(Salpha_overall);
x    = mean(MAE_overall);

% ------------------------------ plot setting
plot(x,y,'o','LineWidth',3);
set(gca,'FontSize',15);


mu = log10([1.095,1.12,1.15,1.21,1.585,1.9954]);
set(gca, 'XTick', mu) 
set(gca,'XTickLabel',{'0.04','0.06','0.08','0.1','0.2','0.3'}) 
axis([0.03 0.31 0.55,0.92])
xlabel('MAE','FontSize',15);
ylabel('S_{\alpha}','FontSize',15);

% ------------------------------ add methods' title 
for i=1:length(y)
    text(x(i),y(i),['  ' num2str(evaluation_methods{i})],'fontsize',14)
end
grid on;

% ------------------------------ sub-figure
xx = [x(18:19),x(21:end)];
yy = [y(18:19),y(21:end)];

axes('Position',[0.16,0.22,0.11,0.31]); % g                                                                     
plot(xx,yy,'o','LineWidth',3);

for j = 1:length(xx)
    text(xx(j),yy(j),['  ' num2str(evaluation_methods_sub{j})],'fontsize',14)  
end

