%% 这是DSO的基准，只包含基线，没有可行域模型，只能用于ED_withoutADS,生成影子价格用
clear,clc;
global deltaT T;
deltaT = 0.25;
T = 24/deltaT;
load('新能源和负荷数据/BL_samples.mat');
base_load = mean(BL_samples,2);
T0 = length(base_load(:,1));
rated_load = base_load(1:T0/T:T0,1);

%% 构建Agg
aggcase = aggcase_medium;
templete_id  = 1;
aggdata = aggcase.data;
Nsamples = 1;
Aggs = {};
i=1;
Aggs{i} = Aggregator(aggdata(i,1),aggdata(i,2),aggdata(i,3),aggdata(i,4),aggdata(i,5),Nsamples);
save Aggs_T96_data Aggs
% 构建DS
% busTS = 5;
% DS = DSO(Aggs,case33modified,rated_load,busTS,templete_id);
% 
% save DS_indiv_info DS


