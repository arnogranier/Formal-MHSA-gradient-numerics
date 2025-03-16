using Random;
using ForwardDiff;
using LinearAlgebra;

Random.seed!(12345);

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
Δ(i,j,h) = Wo[h,:,:]'*e(i)*x[j]';
Γq(i,j,jprime,h) = (dϕ.(q(i,h)) .* ϕ.(k(jprime, h)))*v(j,h)'*Δ(i,i,h);
Γk(i,j,jprime,h) = (ϕ.(q(i,h)) .* dϕ.(k(jprime, h)))*v(j,h)'*Δ(i,jprime,h);
e(i) = 0.5*(y[i] - sum([Wo[h,:,:]*vtilde(i,h) for h in 1:10]));

function E(y, x, Wo, Wv, Wk, Wq)
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
end;

function perturbed(W, ϵ)
  pW = copy(W)
  pW[h,1,1] += ϵ
  return pW
end;

h = 1;
ϵ = 1e-7;

dnumeric = (E(y,x,perturbed(Wo, ϵ),Wv,Wk,Wq)-E(y,x,Wo,Wv,Wk,Wq))/ϵ
danalytic = -sum([e(i)*vtilde(i,h)' for i in 1:10])[1,1]
println("Relative error between finite differences and analytic gradient for one term of W_O : ", abs((danalytic-dnumeric)/dnumeric))

dnumeric = (E(y,x,Wo,Wv,Wk,perturbed(Wq,ϵ))-E(y,x,Wo,Wv,Wk,Wq))/ϵ
danalytic = -sum([(Γq(i,j,j,h) - α(i,j,h)*sum([Γq(i,j,jprime,h) for jprime in 1:10]))/Z(i,h) for j in 1:10 for i in 1:10])[1,1]
println("Relative error between finite differences and analytic gradient for one term of W_Q : ", abs((danalytic-dnumeric)/dnumeric))

dnumeric = (E(y,x,Wo,Wv,perturbed(Wk,ϵ),Wq)-E(y,x,Wo,Wv,Wk,Wq))/ϵ
danalytic = -sum([(Γk(i,j,j,h) - α(i,j,h)*sum([Γk(i,j,jprime,h) for jprime in 1:10]))/Z(i,h) for j in 1:10 for i in 1:10])[1,1]
println("Relative error between finite differences and analytic gradient for one term of W_K : ", abs((danalytic-dnumeric)/dnumeric))

dnumeric = (E(y,x,Wo,perturbed(Wv,ϵ),Wk,Wq)-E(y,x,Wo,Wv,Wk,Wq))/ϵ
danalytic = -sum([α(i,j,h)*Δ(i,j,h) for j in 1:10 for i in 1:10])[1,1]
println("Relative error between finite differences and analytic gradient for one term of W_V : ", abs((danalytic-dnumeric)/dnumeric))

dauto = ForwardDiff.gradient(Wo->E(y,x,Wo,Wv,Wk,Wq), Wo)
danalytic = [-sum([e(i)*vtilde(i,h)' for i in 1:10]) for h in 1:10]
println("Mean absolute error between forward automatic differentiation and analytic gradient for W_O : ", sum(dauto-permutedims(cat(danalytic...,dims=3),(3,1,2)))/length(dauto))

dauto = ForwardDiff.gradient(Wq->E(y,x,Wo,Wv,Wk,Wq), Wq)
danalytic = [-sum([(Γq(i,j,j,h) - α(i,j,h)*sum([Γq(i,j,jprime,h) for jprime in 1:10]))/Z(i,h) for j in 1:10 for i in 1:10]) for h in 1:10]
println("Mean absolute error between forward automatic differentiation and analytic gradient for W_Q : ", sum(dauto-permutedims(cat(danalytic...,dims=3),(3,1,2)))/length(dauto))

dauto = ForwardDiff.gradient(Wk->E(y,x,Wo,Wv,Wk,Wq), Wk)
danalytic = [-sum([(Γk(i,j,j,h) - α(i,j,h)*sum([Γk(i,j,jprime,h) for jprime in 1:10]))/Z(i,h) for j in 1:10 for i in 1:10]) for h in 1:10]
println("Mean absolute error between forward automatic differentiation and analytic gradient for W_K : ", sum(dauto-permutedims(cat(danalytic...,dims=3),(3,1,2)))/length(dauto))

dauto = ForwardDiff.gradient(Wv->E(y,x,Wo,Wv,Wk,Wq), Wv)
danalytic = [-sum([α(i,j,h)*Δ(i,j,h) for j in 1:10 for i in 1:10]) for h in 1:10]
println("Mean absolute error between forward automatic differentiation and analytic gradient for W_V : ", sum(dauto-permutedims(cat(danalytic...,dims=3),(3,1,2)))/length(dauto))