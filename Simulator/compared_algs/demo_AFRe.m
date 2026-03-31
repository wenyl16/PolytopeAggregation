load('Aggs_data_1.mat')
global deltaT T;    
deltaT = 2;
T = 24/deltaT;
[A,b]=cal_AFRe(Aggs{1, 1}.bounds.pmin,Aggs{1, 1}.bounds.pmax,Aggs{1, 1}.bounds.emin,Aggs{1, 1}.bounds.emax);