% clear,clc;
% global deltaT T;    
% deltaT = 1;
% T = 24/deltaT;
% load('Aggs_data.mat')
% load('boundaries.mat')
% boundaries.pmax = pmax;
% boundaries.emax = emax;
% boundaries.pmin = pmin;
% boundaries.emin = emin;
% 
% templete_id = 3;
% for i = 1:size(Aggs,2)
%     Aggs{i}.model = [];
%     Aggs{i} = Aggs{i}.model_agg_with_boundaries(templete_id, boundaries);
% end
% save Aggs_data_wang Aggs

% clear,clc;
% global deltaT T;    
% deltaT = 2;
% T = 24/deltaT;
% load('Aggs_data.mat')
% templete_id = 4;
% for i = 1:size(Aggs,2)
%     Aggs{i}.model = [];
%     Aggs{i} = Aggs{i}.model_agg(templete_id);
% end
% save Aggs_data_box Aggs

clear,clc;
global deltaT T;    
deltaT = 1;
T = 24/deltaT;
load('Aggs_data.mat')
load('boundaries.mat')
boundaries.pmax = pmax;
boundaries.emax = emax;
boundaries.pmin = pmin;
boundaries.emin = emin;

templete_id = 1;
for i = 1:size(Aggs,2)
    Aggs{i}.model = [];
    Aggs{i} = Aggs{i}.model_agg_with_boundaries(templete_id, boundaries);
end
polytope.A = Aggs{1}.model.A*cumsum(eye(T))*diag(sum(boundaries.pmax,2)-sum(boundaries.pmin,2));
polytope.b = Aggs{1}.model.b - Aggs{1}.model.A*cumsum(eye(T))*sum(boundaries.pmin,2);

save polytope_params polytope
%save Aggs_T24_data_1 Aggs

% clear,clc;
% global deltaT T;
% deltaT = 2;
% T = 24/deltaT;
% templete_id = 2;
% load('Aggs_data.mat')
% for i = 1:size(Aggs,2)
%     Aggs{i}.model = [];
%     Aggs{i} = Aggs{i}.model_agg(templete_id);
% end
% save Aggs_data_2 Aggs
