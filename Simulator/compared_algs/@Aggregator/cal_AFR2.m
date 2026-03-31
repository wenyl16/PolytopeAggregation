function [A,b]=cal_AFR2(obj,pmin,pmax,emin,emax)
    [Q,N] = size(pmin);
    V = eye(Q);
    phi_max = sum(emax,2);
    phi_min = sum(emin,2);
    temp = nchoosek(1:Q,2);
    x = temp(:,1);
    y = temp(:,2);
    for k = 1:length(x)
        v = zeros(1,Q);
        v(y(k))=1;v(x(k))=-1;
        V = [V;v];
        phi_max(Q+k)=0;
        phi_min(Q+k)=0;
        for i = 1:N
            phi_max(Q+k)=phi_max(Q+k)+min(emax(y(k),i)-emin(x(k),i),sum(pmax(x(k)+1:y(k),i)));
            phi_min(Q+k)=phi_min(Q+k)+max(emin(y(k),i)-emax(x(k),i),sum(pmin(x(k)+1:y(k),i)));
        end
    end
%     U = cumsum((V*flip(eye(Q)))')'*flip(eye(Q));
%     A = [U;-U];
    A = [V;-V];
    b = [phi_max;-phi_min];
end