function [A,b]=cal_AFRbox(obj,bounds)
    pmin = bounds.pmin;
    pmax = bounds.pmax;
    emin = bounds.emin;
    emax = bounds.emax;
    [Q,N] = size(pmin);    
    vp_up = sdpvar(Q,N,'full');
    vp_down = sdpvar(Q,N,'full');
    constraints = [
        pmin<=vp_up<=pmax,...
        emin<=cumsum(vp_up)<=emax,...
        pmin<=vp_down<=pmax,...
        emin<=cumsum(vp_down)<=emax,...
        vp_up>=vp_down,...
    ];
    f = -sum(sum(vp_up,2)-sum(vp_down,2));
    ops = sdpsettings('verbose',2,'solver','cplex');
    optimize(constraints,f,ops);
    Pmin = sum(value(vp_down),2);
    Pmax = sum(value(vp_up),2);
    
    A = [eye(Q);-eye(Q)]*cumsum(eye(Q))^(-1);
    b = [Pmax;-Pmin];
end