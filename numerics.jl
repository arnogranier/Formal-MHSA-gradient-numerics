using Random;
using LinearAlgebra;

x = [2*rand(10) .- 1 for i in 1:10];
y = [2*rand(10) .- 1 for i in 1:10];
Wk = 2*rand(10,10,10) .- 1;
Wq = 2*rand(10,10,10) .- 1;
Wv = 2*rand(10,10,10) .- 1;
Wo = 2*rand(10,10,10) .- 1;

ϕ=sin;
dϕ=cos;

k(i,h) = Wk[h,:,:] * x[i];
q(i,h) = Wq[h,:,:] * x[i];
v(i,h) = Wv[h,:,:] * x[i];
Z(i,h) = sum([dot(ϕ.(q(i,h)), ϕ.(k(j,h))) for j in 1:10]);
α(i,j,h) = dot(ϕ.(q(i,h)),ϕ.(k(j,h))) / Z(i,h);
vtilde(i,h) = sum([α(i,j,h)*v(j,h) for j in 1:10]);
Δ(i,j,h) = Wo[h,:,:]'*e[i]*x[j]';
Γq(i,j,jprime,h) = (dϕ.(q(i,h)) .* ϕ.(k(jprime, h)))*v(j,h)'*Δ(i,i,h);
Γk(i,j,jprime,h) = (ϕ.(q(i,h)) .* dϕ.(k(jprime, h)))*v(j,h)'*Δ(i,jprime,h);

function E(y)
    p = 0.5*[y[i] - sum([Wo[h,:,:]*vtilde(i,h) for h in 1:10]) for i in 1:10]
    sum(dot(t,t) for t in p), p
end;

h = 1;
ϵ = 1e-7;

E0, e = E(y);
Wo[h,1,1] += ϵ;
E1, _ = E(y);
dnumeric = (E1-E0)/ϵ
danalytic = -sum([e[i]*vtilde(i,h)' for i in 1:10])[1,1]

E0, e = E(y);
Wq[h,1,1] += ϵ;
E1, _ = E(y);
dnumeric = (E1-E0)/ϵ
danalytic = -sum([(Γq(i,j,j,h) - α(i,j,h)*sum([Γq(i,j,jprime,h) for jprime in 1:10]))/Z(i,1) for j in 1:10 for i in 1:10])[1,1]

E0, e = E(y);
Wk[h,1,1] += ϵ;
E1, _ = E(y);
dnumeric = (E1-E0)/ϵ
danalytic = -sum([(Γk(i,j,j,h) - α(i,j,h)*sum([Γk(i,j,jprime,h) for jprime in 1:10]))/Z(i,h) for j in 1:10 for i in 1:10])[1,1]

E0, e = E(y);
Wv[h,1,1] += ϵ;
E1, _ = E(y);
dnumeric = (E1-E0)/ϵ
danalytic = -sum([α(i,j,h)*Δ(i,j,h) for j in 1:10 for i in 1:10])[1,1]