using StochasticDiffEq, DiffEqDevTools, Test

linear = (u,p,t) -> (p*u)
g = (u,p,t) -> zero(u)
linear_analytic = (u0,p,t,W) -> u0*exp(p*t)
prob = SDEProblem(SDEFunction(linear,g,analytic=linear_analytic),g,
                  1/2,(0.0,1.0),1.01)

dts = (1/2) .^ (7:-1:4) #14->7 good plot
sol = solve(prob,SKenCarp())
sim2 = test_convergence(dts,prob,SKenCarp(),trajectories=20)
@test abs(sim2.𝒪est[:l∞]-3) <.1 #High tolerance since low dts for testing!

linear = (du,u,p,t) -> (du.=p.*u)
g = (du,u,p,t) -> (du.=0)
linear_analytic = (u0,p,t,W) -> u0*exp.(p.*t)
prob = SDEProblem(SDEFunction(linear,g,analytic=linear_analytic),g,
                  rand(4,2),(0.0,1.0),1.01)

dts = (1/2) .^ (7:-1:4) #14->7 good plot
sol = solve(prob,SKenCarp())
sim2 = test_convergence(dts,prob,SKenCarp(),trajectories=20)
@test abs(sim2.𝒪est[:l∞]-3) <.1 #High tolerance since low dts for testing!



# Stochastic Runge Kutta for the weak approximation (deterministic order tests)
linear = (u,p,t) -> (p*u)
g = (u,p,t) -> zero(u)
linear_analytic = (u0,p,t,W) -> u0*exp(p*t)
prob = SDEProblem(SDEFunction(linear,g,analytic=linear_analytic),g, 1/2,(0.0,1.0),1.01)
dts = 1 .//2 .^(10:-1:2)

sim2 = test_convergence(dts,prob, DRI1(),trajectories=20)
@test abs(sim2.𝒪est[:l∞]-3) <.1
sim2 = test_convergence(dts,prob, RI1(),trajectories=20)
@test abs(sim2.𝒪est[:l∞]-3) <.1
