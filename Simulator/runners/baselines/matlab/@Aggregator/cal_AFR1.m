function [A,b]=cal_AFR1(obj,pmin,pmax,emin,emax)
    [Q,N] = size(pmin);
    Pmin = sum(pmin,2);
    Pmax = sum(pmax,2);
    Emin = sum(emin,2);
    Emax = sum(emax,2);
    A = [eye(Q);-eye(Q);cumsum(eye(Q));-cumsum(eye(Q))]*cumsum(eye(Q))^(-1);
    b = [Pmax;-Pmin;Emax;-Emin];
%     A([1,Q+1],:)=[];
%     b([1,Q+1]) = [];
end