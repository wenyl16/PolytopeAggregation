clear,clc;
global deltaT T;
deltaT = 2;
T = 24/deltaT;
ops = sdpsettings('verbose',2,'solver','cplex');
U = [zeros(1,T)];
ntest = 50;

rate1 = [];
rate2 = [];
rate_box = [];
rate_wang = [];
load('Aggs_data_1.mat');
Agg_1 = Aggs{1};
load('Aggs_data_2.mat');
Agg_2 = Aggs{1};
load('Aggs_data_box.mat');
Agg_box = Aggs{1};
load('Aggs_data_wang.mat');
Agg_wang = Aggs{1};
x = sdpvar(T,size(Agg_1.bounds.pmin,2),'full');
P = sdpvar(T,1,'full');
cons_ext = [sum(x,2)==P,...
    Agg_1.bounds.pmin<=x<=Agg_1.bounds.pmax,....
    Agg_1.bounds.emin<=cumsum(x)<=Agg_1.bounds.emax];

cons_1 = [Agg_1.model.A*cumsum(P)<=Agg_1.model.b];
cons_2 = [Agg_2.model.A*cumsum(P)<=Agg_2.model.b];
cons_box = [Agg_box.model.A*cumsum(P)<=Agg_box.model.b];
cons_wang = [Agg_wang.model.A*cumsum(P)<=Agg_wang.model.b];
for i = 1:ntest
    while 1
    u = round(rand(T,1));
        if isempty(find(sum(abs(u'-U),2)==0))
            U = [U;u'];
            break
        end
    end
    f = u'*P;
    optimize(cons_ext,f,ops);
    f_min_ext = value(f);
    optimize(cons_ext,-f,ops);
    f_max_ext = value(f);
    
    optimize(cons_1,f,ops);
    f_min_1 = value(f);
    optimize(cons_1,-f,ops);
    f_max_1 = value(f);
    
    optimize(cons_2,f,ops);
    f_min_2 = value(f);
    optimize(cons_2,-f,ops);
    f_max_2 = value(f);
    
    optimize(cons_box,f,ops);
    f_min_box = value(f);
    optimize(cons_box,-f,ops);
    f_max_box = value(f);
    
    optimize(cons_wang,f,ops);
    f_min_wang = value(f);
    optimize(cons_wang,-f,ops);
    f_max_wang = value(f);
    
    rate1 = [rate1, (f_max_1-f_min_1)/(f_max_ext-f_min_ext)];
    rate2 = [rate2, (f_max_2-f_min_2)/(f_max_ext-f_min_ext)];
    rate_box = [rate_box, (f_max_box-f_min_box)/(f_max_ext-f_min_ext)];
    rate_wang = [rate_wang, (f_max_wang-f_min_wang)/(f_max_ext-f_min_ext)];
end
rate1_mean = geo_mean(rate1);
rate2_mean = geo_mean(rate2);
rate_box(rate_box==0) = [];
rate_box_mean = geo_mean(rate_box);
rate_wang_mean = geo_mean(rate_wang);

% rate1_mean1 = mean(rate1);
% rate2_mean1 = mean(rate2);
% rate_box_mean1 = mean(rate_box);
% rate_wang_mean1 = mean(rate_wang);
