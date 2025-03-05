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
Δ(i,j,h) = Wo[h,:,:]'*e0[i]*x[j]';
Γq(i,j,jprime,h) = (dϕ.(q(i,h)) .* ϕ.(k(jprime, h)))*v(j,h)'*Δ(i,i,h);
Γk(i,j,jprime,h) = (ϕ.(q(i,h)) .* dϕ.(k(jprime, h)))*v(j,h)'*Δ(i,jprime,h);
e(y) = 0.5*[y[i] - sum([Wo[h,:,:]*vtilde(i,h) for h in 1:10]) for i in 1:10]

function E(y)
    p = 0.5*[y[i] - sum([Wo[h,:,:]*vtilde(i,h) for h in 1:10]) for i in 1:10]
    sum(dot(t,t) for t in p)
end;

function E_explicit(y, x, Wo, Wv, Wk, Wq)
  xtilde = [zeros(Real,10) for i in 1:10];
  for i in 1:10
    for h in 1:10
      vtilde = zeros(Real, 10)
      Z = sum([dot(ϕ.(Wq[h,:,:] * x[i]), ϕ.(Wk[h,:,:] * x[j])) for j in 1:10]);
      q = Wq[h,:,:] * x[i]
      for j in 1:10
        k = Wk[h,:,:] * x[j]
        v = Wv[h,:,:] * x[j]
        alpha = dot(ϕ.(k),ϕ.(q))/Z
        vtilde += alpha * v
      end
    xtilde[i] .= xtilde[i]+Wo[h,:,:]*vtilde
    end
  end
  p = 0.5*[y[i] - xtilde[i] for i in 1:10]
  return sum(dot(t,t) for t in p)
end

E(y)
E_explicit(y,x,Wo,Wv,Wk,Wq)

h = 1;
ϵ = 1e-7;

E0, e0 = E(y), e(y);
Wo[h,1,1] += ϵ;
E1 = E(y);
dnumeric = (E1-E0)/ϵ
danalytic = -sum([e0[i]*vtilde(i,h)' for i in 1:10])[1,1]

E0, e0 = E(y), e(y);
Wq[h,1,1] += ϵ;
E1 = E(y);
dnumeric = (E1-E0)/ϵ
danalytic = -sum([(Γq(i,j,j,h) - α(i,j,h)*sum([Γq(i,j,jprime,h) for jprime in 1:10]))/Z(i,h) for j in 1:10 for i in 1:10])[1,1]

E0, e0 = E(y), e(y);
Wk[h,1,1] += ϵ;
E1 = E(y);
dnumeric = (E1-E0)/ϵ
danalytic = -sum([(Γk(i,j,j,h) - α(i,j,h)*sum([Γk(i,j,jprime,h) for jprime in 1:10]))/Z(i,h) for j in 1:10 for i in 1:10])[1,1]

E0, e0 = E(y), e(y);
Wv[h,1,1] += ϵ;
E1 = E(y);
dnumeric = (E1-E0)/ϵ
danalytic = -sum([α(i,j,h)*Δ(i,j,h) for j in 1:10 for i in 1:10])[1,1]

using ForwardDiff

dauto = ForwardDiff.gradient(Wo->E_explicit(y,x,Wo,Wv,Wk,Wq), Wo)
danalytic = [-sum([e0[i]*vtilde(i,h)' for i in 1:10]) for h in 1:10]
sum(dauto-permutedims(cat(danalytic...,dims=3),(3,1,2)))/length(dauto)

dauto = ForwardDiff.gradient(Wv->E_explicit(y,x,Wo,Wv,Wk,Wq), Wv)
danalytic = [-sum([α(i,j,h)*Δ(i,j,h) for j in 1:10 for i in 1:10]) for h in 1:10]
sum(dauto-permutedims(cat(danalytic...,dims=3),(3,1,2)))/length(dauto)

dauto = ForwardDiff.gradient(Wk->E_explicit(y,x,Wo,Wv,Wk,Wq), Wk)
danalytic = [-sum([(Γk(i,j,j,h) - α(i,j,h)*sum([Γk(i,j,jprime,h) for jprime in 1:10]))/Z(i,h) for j in 1:10 for i in 1:10]) for h in 1:10]
sum(dauto-permutedims(cat(danalytic...,dims=3),(3,1,2)))/length(dauto)

dauto = ForwardDiff.gradient(Wq->E_explicit(y,x,Wo,Wv,Wk,Wq), Wq)
danalytic = [-sum([(Γq(i,j,j,h) - α(i,j,h)*sum([Γq(i,j,jprime,h) for jprime in 1:10]))/Z(i,h) for j in 1:10 for i in 1:10]) for h in 1:10]
sum(dauto-permutedims(cat(danalytic...,dims=3),(3,1,2)))/length(dauto)
