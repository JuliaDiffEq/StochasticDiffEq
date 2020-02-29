using StochasticDiffEq, DiffEqDevTools, LinearAlgebra, Random, Test
Random.seed!(100)
dts = (1/2) .^ (10:-1:2) #14->7 good plot

# From RUNGE–KUTTA METHODS FOR THE STRONG APPROXIMATION OF SOLUTIONS OF STOCHASTIC DIFFERENTIAL EQUATIONS
# (7.4)

d = 4; m = 10
u0 = rand(4)
A = zeros(d,d)
for i in 1:d, j in 1:d
  global A
  i == j && (A[i,j] = -3/2)
  i != j && (A[i,j] = 1/20)
end

B = [zeros(d,d) for i in 1:m]
for k in 1:m, i in 1:d, j in 1:d
  global B
  i == j && (B[k][i,j] = 1/5)
  i != j && (B[k][i,j] =1/100)
end

function f_noncommute(du,u,p,t)
  mul!(du,A,u)
end

function g_noncommute(du,u,p,t)
  for i in 1:m
    mul!(@view(du[:,i]),B[i],u)
  end
end

function f_noncommute_analytic(u0,p,t,W)
  tmp = (A - 0.5*sum(B[i]^2 for i in 1:m))*t + sum(B[i]*W[i] for i in 1:m)
  exp(tmp)*u0
end

ff_noncommute = SDEFunction(f_noncommute,g_noncommute,analytic=f_noncommute_analytic)

prob = SDEProblem(ff_noncommute,g_noncommute,u0,(0.0,1.0),noise_rate_prototype=rand(4,m))

sol = solve(prob,EM(),dt=1/2^(8))
sol = solve(prob,RKMil_General(),dt=1/2^(8))

dts = (1/2) .^ (10:-1:3) #14->7 good plot
sim1 = test_convergence(dts,prob,EM(),trajectories=Int(1e2))
@test abs(sim1.𝒪est[:final] - 0.5) < 0.2
sim2 = test_convergence(dts,prob,RKMilCommute(),trajectories=Int(1e2))
@test abs(sim2.𝒪est[:final] - 1) < 0.2
sim3 = test_convergence(dts,prob,RKMil_General(),trajectories=Int(1e2))
@test abs(sim2.𝒪est[:final] - 1) < 0.2

d = 2; m = 4
u0 = [2.0,2.0]
α = 1/2

function f_noncommute(du,u,p,t)
  du .= 0
end

function g_noncommute(du,u,p,t)
  du[1,1] = cos(α)*sin(u[1])
  du[2,1] = sin(α)*sin(u[1])
  du[1,2] = cos(α)*cos(u[1])
  du[2,2] = sin(α)*cos(u[1])
  du[1,3] = -sin(α)*sin(u[2])
  du[2,3] = cos(α)*sin(u[2])
  du[1,4] = -sin(α)*cos(u[2])
  du[2,4] = cos(α)*cos(u[2])
end

prob = SDEProblem(f_noncommute,g_noncommute,u0,(0.0,1.0),noise_rate_prototype=rand(2,m))

sol1 = solve(prob,EM(),dt=1/2^(8))
sol2 = solve(prob,RKMil_General(),dt=1/2^(8),adaptive=false)
sol3 = solve(prob,RKMil_General(),dt=1/2^(8))

dts = (1/2) .^ (7:-1:2) #14->7 good plot
test_dt = 1/2 ^ (8)
sim1 = analyticless_test_convergence(dts,prob,EM(),test_dt,trajectories=400)
@test abs(sim1.𝒪est[:final] - 0.5) < 0.2
sim2 = analyticless_test_convergence(dts,prob,RKMil_General(),test_dt,trajectories=400)
@test_broken abs(sim1.𝒪est[:final] - 1) < 0.2
