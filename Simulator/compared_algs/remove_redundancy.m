clear,clc;
load Aggs_data_1;
b = Aggs{1, 1}.model.b;
A = Aggs{1, 1}.model.A;
[Nc,T] = size(A);
deltaT = 24/T;
ops = sdpsettings('verbose',2,'solver','cplex');
P = sdpvar(T,1,'full');
cons = [A*cumsum(P)*deltaT<=b];
for i = 1:Nc
    f = -A(i,:)*cumsum(P)*deltaT;
    optimize(cons,f,ops);
    b(i) = -value(f);
end

save b_without_redundacy b;
