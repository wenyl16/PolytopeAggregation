function [A,b,solvertime]=cal_AFRwang(obj,bounds)
    global T deltaT;
    N = size(bounds.pmin,2);
    cal_AFR = @obj.cal_AFR1;

    pmin = bounds.pmin;
    pmax = bounds.pmax;
    emin = bounds.emin;
    emax = bounds.emax;
    [A,b]=cal_AFR(pmin,pmax,emin,emax);
    b0 = b;
    ops = sdpsettings('solver','cplex','verbose',2);

    C = sparse(repmat(eye(T),1,N));
    D = sparse(4*N*T,N*T);
    for i = 1:N
        D(4*T*(i-1)+1:4*T*i,T*(i-1)+1:T*i) = [eye(T);-eye(T);cumsum(eye(T));-cumsum(eye(T))];
    end
    e = reshape([pmax;-pmin;emax;-emin],4*N*T,1);
    ops = sdpsettings('solver','cplex','verbose',2);
    %M = 10*max(abs(b(3*T)),abs(b(4*T)));
    M = 100*max(abs(b));
%     W = sparse([eye(T-1),eye(T-1),zeros(T-1,T),zeros(T-1,T);
%         zeros(T,T-1),zeros(T,T-1),eye(T),eye(T);]);
    W = sparse([eye(T),eye(T),zeros(T,T),zeros(T,T);
        zeros(T),zeros(T),eye(T),eye(T);]);
    n = length(b);
    P1 = sdpvar(T,1,'full');
    omega = sdpvar(n,1,'full');
    s = binvar(n,1,'full');
    pai = sdpvar(T,1,'full');
    lambda = sdpvar(4*N*T,1,'full');
    x = sdpvar(T*N,1,'full');
    fs = [10];
    solvertime = 0;
    for k = 1:40
        cons = [
          omega'*A*cumsum(eye(T))+pai'==0,...
          lambda'*D+pai'*C==0,...
          lambda>=0,...
          0<=b-A*cumsum(eye(T))*P1<=M*(1-s),...
          0<=omega<=M*s,...
          W*s<=1,...
          sum(s)>=T
        ];
        f = lambda'*e-omega'*b;
        res = optimize(cons ,f,ops);
        solvertime = solvertime+res.solvertime;
        fs = [fs,value(-f)];
        %value(pai)

        P0=sdpvar(T,1,'full');
        cons_0 = [
            C*x-P0==0,...
            D*x<=e,...
        ];
        obj_0 = norm(P0-value(P1));
        %obj_0 = -value(mu)'*P0;
        res = optimize(cons_0,obj_0,ops);
        solvertime = solvertime+res.solvertime;
        value(obj_0)
        if value(obj_0)<=1e-4
            break;
        end
        i_b_act = A*value(cumsum(P1))>=b-1e-6;
        b_var = sdpvar(sum(i_b_act),1,'full');
        P_var = sdpvar(T,1,'full');
        z = binvar(sum(i_b_act),1,'full');
        cons_1 = [
            b_var<=b(i_b_act),...
            A(i_b_act,:)*value(cumsum(P0))>=b_var-M*(1-z),...
            sum(z)>=T,...
            A(i_b_act,:)*cumsum(P_var)<=b_var,...
            ];
        obj_1 = -sum(b_var);%+1e-5*b_var(i_b_act)'*b_var(i_b_act);
        res = optimize(cons_1,obj_1,ops);
        solvertime = solvertime +res.solvertime;

        b(i_b_act) = value(b_var);
    end
end


