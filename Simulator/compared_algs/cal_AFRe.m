function [A,b]=cal_AFRe(pmin,pmax,emin,emax)
[Q,N] = size(pmin);
U = ff2n(Q);
U = U(2:end,:);
V = U(:,1:end)-[zeros(2^Q-1,1),U(:,1:end-1)];
phi_max = zeros(2^Q-1,1);
phi_min = zeros(2^Q-1,1);
K1=0;K2=0;
tic
for i = 1:2^Q-1
    u=U(i,:);
    v = u-[0,u(1:end-1)];
    qs = find (v,1,'first')-1;
    qt = find (v,1,'last')-1;
    if u(end)==1
        K1=K1+1;
        for k = 1:N
            phi_max(i) = phi_max(i)+min(emax(Q-qs,k)-(ones(1,qt-qs)-u(qs+1:qt))*pmin(Q-qs:-1:Q-qt+1,k),...
                emax(Q-qt,k)+u(qs+1:qt)*pmax(Q-qs:-1:Q-qt+1,k));
            
            phi_min(i) = phi_min(i)+max(emin(Q-qs,k)-(ones(1,qt-qs)-u(qs+1:qt))*pmax(Q-qs:-1:Q-qt+1,k),...
                emin(Q-qt,k)+u(qs+1:qt)*pmin(Q-qs:-1:Q-qt+1,k));
        end
    else
        K2=K2+1;
        for k = 1:N
            phi_max(i) = phi_max(i)+min(emax(Q-qs,k)-emin(Q-qt,k)-...
                (ones(1,qt-qs-1)-u(qs+2:qt))*pmin(Q-qs-1:-1:Q-qt+1,k),...
                +u(qs+1:qt)*pmax(Q-qs:-1:Q-qt+1,k));
            
            phi_min(i) = phi_min(i)+max(emin(Q-qs,k)-emax(Q-qt,k)-...
                (ones(1,qt-qs-1)-u(qs+2:qt))*pmax(Q-qs-1:-1:Q-qt+1,k),...
                +u(qs+1:qt)*pmin(Q-qs:-1:Q-qt+1,k));
        end
    end
end
toc
A=[V;-V]*flip(eye(Q));
b = [phi_max;-phi_min];
end