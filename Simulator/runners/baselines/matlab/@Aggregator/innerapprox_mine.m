function [A,b,fs,solvertime]=innerapprox_mine(obj,templete_id,bounds)
    global T deltaT;
    N = size(bounds.pmin,2);
    if templete_id == 1
        cal_AFR = @obj.cal_AFR1;
    elseif templete_id ==2
        cal_AFR = @obj.cal_AFR2;
    end
    pmin = bounds.pmin;
    pmax = bounds.pmax;
    emin = bounds.emin;
    emax = bounds.emax;
    [A,b]=cal_AFR(pmin,pmax,emin,emax);
    b0 = b;
    ops = sdpsettings('solver','cplex','verbose',2);
    %options.cplex.mip.tolerances.mipgap=1e-1; 
    C = sparse(repmat(eye(T),1,N));
    D = sparse(4*N*T,N*T);
    for i = 1:N
        D(4*T*(i-1)+1:4*T*i,T*(i-1)+1:T*i) = [eye(T);-eye(T);cumsum(eye(T));-cumsum(eye(T))];
    end
    e = reshape([pmax;-pmin;emax;-emin],4*N*T,1);

    %M = 10*max(abs(b(3*T)),abs(b(4*T)));
    M = 1000*max(abs(b));
    x = sdpvar(T*N,1,'full');
    P1 = sdpvar(T,1,'full');
    y = sdpvar(T,1,'full');
    u = binvar(T,1,'full');
    omega = sdpvar(T,1,'full');
    lambda = sdpvar(4*N*T,1,'full');
    fs = [10];
    solvertime = 0;
    for k = 1:20
        %上界
        cons = [
        -M*u<=y<=M*u,...
        -M*(1-u)<=y+P1<=M*(1-u),...
          A*cumsum(P1)<=b,...
          u + omega ==0,...
          C'*omega+D'*lambda==0,...
          lambda>=0,...
        ];
        f1 = sum(y);
        f2 = e'*lambda;
        f = f1+f2;
        res = optimize(cons ,f,ops);
        solvertime = solvertime +res.solvertime;
        %value(mu)
%         if norm(value(u))~=0
%             upper_error = value(-f)/norm(value(u));
%         else
%             upper_error = 0;
%         end
        upper_error = value(-f);
        if upper_error>1e-4
            P0=sdpvar(T,1,'full');
            cons_0 = [
                C*x-P0==0,...
                D*x<=e,...
            ];
            obj_0 = norm(P0-value(P1));
            %obj_0 = -value(mu)'*P0;
            optimize(cons_0,obj_0,ops);
            value(obj_0)

            i_b_act = A*value(cumsum(P1))>b-1e-5;
            %i_b_act = A*value(cumsum(P1))>=b-1e-8;
            b_var = sdpvar(size(b,1),1,'full');
            P_var = sdpvar(T,1,'full');
            z = binvar(sum(i_b_act),1,'full');
            cons_1 = [
                b_var<=b,...
                A(i_b_act,:)*value(cumsum(P0))>=b_var(i_b_act)-M*(1-z),...
                sum(z)>=T,...
                A*cumsum(P_var)<=b_var,...
                ];
            obj_1 = -sum(b_var);%+1e-5*b_var(i_b_act)'*b_var(i_b_act);
            res = optimize(cons_1,obj_1,ops);

            b = value(b_var);
        end
        %下界
        cons = [
          -M*u<=y<=M*u,...
          -M*(1-u)<=y-P1<=M*(1-u),...
          A*cumsum(P1)<=b,...
          -u + omega ==0,...
          C'*omega+D'*lambda==0,...
          lambda>=0,...
        ];
        f1 = sum(y);
        f2 = e'*lambda;
        f = f1+f2;
        res = optimize(cons ,f,ops);
        solvertime = solvertime +res.solvertime;
        %value(mu)
%         if norm(value(u))~=0
%             lower_error = value(-f)/norm(value(u));
%         else
%             lower_error = 0;
%         end
        lower_error = value(-f);
        if lower_error>1e-4

        P0=sdpvar(T,1,'full');
        cons_0 = [
            C*x-P0==0,...
            D*x<=e,...
        ];
        obj_0 = norm(P0-value(P1));
        %obj_0 = -value(mu)'*P0;
        res=optimize(cons_0,obj_0,ops);
        solvertime = solvertime +res.solvertime;
        %value(obj_0)

        %i_b_act = A*value(cumsum(P1))>=b-1e-8;
        i_b_act = A*value(cumsum(P1))>=b-1e-5;
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
        fs = [fs,max(upper_error,lower_error)];

        if max(upper_error,lower_error)<=1e-4
            break;
        end
    end
end