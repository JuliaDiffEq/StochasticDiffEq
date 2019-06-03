@muladd function perform_step!(integrator,cache::SROCK_1ConstantCache,f=integrator.f)
  @unpack t,dt,uprev,u,W,p = integrator

  maxeig!(integrator, cache)
  mdeg = Int(floor(sqrt(2*dt*integrator.eigen_est/0.21)+1))
  cache.mdeg = max(3,min(mdeg,250))
  choose_deg!(integrator,cache)

  η  = cache.optimal_η
  ω₀ = 1 + η/(mdeg^2)
  ωSq = ω₀^2 - 1
  Sqrt_ω = sqrt(ωSq)
  cosh_inv = log(ω₀ + Sqrt_ω)             # arcosh(ω₀)
  ω₁ = (sinh(mdeg*cosh_inv)*ωSq)/(cosh(mdeg*cosh_inv)*mdeg*Sqrt_ω - ω₀*sinh(mdeg*cosh_inv))

  α  = cosh(mdeg*cosh_inv)/(2*cosh((mdeg-1)*cosh_inv))
  γ  = 1/(2*α)
  β  = -γ

  uᵢ₋₂ = copy(uprev)
  k = integrator.f(uprev,p,t)
  Tᵢ₋₂ = one(eltype(u))
  Tᵢ₋₁ = ω₀
  Tᵢ   = Tᵢ₋₁
  tᵢ₋₁ = t + dt*(ω₁/ω₀)
  tᵢ   = tᵢ₋₁
  tᵢ₋₂ = t

  #stage 1
  uᵢ₋₁ = uprev + (dt*ω₁/ω₀)*k

  for i in 2:mdeg
    Tᵢ = 2*ω₀*Tᵢ₋₁ - Tᵢ₋₂
    μ   = 2*ω₁*Tᵢ₋₁/Tᵢ
    ν   = 2*ω₀*Tᵢ₋₁/Tᵢ
    κ = - Tᵢ₋₂/Tᵢ
    k = integrator.f(uᵢ₋₁,p,tᵢ₋₁)
    u = dt*μ*k + ν*uᵢ₋₁ + κ*uᵢ₋₂
    (i == mdeg - 1) && (gₘ₋₂ = integrator.g(uᵢ₋₁,p,tᵢ₋₁); u = α*W.dW*gₘ₋₂)
    (i == mdeg) && (gₘ₋₁ = integrator.g(uᵢ₋₁,p,tᵢ₋₁); u = β*W.dW*gₘ₋₂ + γ*W.dW*gₘ₋₁)
    if i < mdeg
      tᵢ = dt*μ + ν*tᵢ₋₁ + κ*tᵢ₋₂
      uᵢ₋₂ = uᵢ₋₁
      uᵢ₋₁ = u_i
      tᵢ₋₂ = tᵢ₋₁
      tᵢ₋₁ = tᵢ
      Tᵢ₋₂ = Tᵢ₋₁
      Tᵢ₋₁ = Tᵢ
    end
  end
  integrator.u = u
end

@muladd function perform_step!(integrator,cache::SROCK_1Cache,f=integrator.f)
  @unpack uᵢ₋₁,k, gₘ₋₁, gₘ₋₂ = cache
  @unpack t,dt,uprev,u,W,p = integrator
  ccache = cache.constantcache
  maxeig!(integrator, cache)
  mdeg = Int(floor(sqrt(2*dt*integrator.eigen_est/0.21)+1))
  ccache.mdeg = max(3,min(mdeg,250))
  choose_deg!(integrator,cache)
  uᵢ₋₂ = cache.tmp

  η  = ccache.optimal_η
  ω₀ = 1 + η/(mdeg^2)
  ωSq = ω₀^2 - 1
  Sqrt_ω = sqrt(ωSq)
  cosh_inv = log(ω₀ + Sqrt_ω)             # arcosh(ω₀)
  ω₁ = (sinh(mdeg*cosh_inv)*ωSq)/(cosh(mdeg*cosh_inv)*mdeg*Sqrt_ω - ω₀*sinh(mdeg*cosh_inv))

  α  = cosh(mdeg*cosh_inv)/(2*cosh((mdeg-1)*cosh_inv))
  γ  = 1/(2*α)
  β  = -γ

  @.. uᵢ₋₂ = uprev
  integrator.f(k,uprev,p,t)
  Tᵢ₋₂ = one(eltype(u))
  Tᵢ₋₁ = ω₀
  Tᵢ   = Tᵢ₋₁
  tᵢ₋₁ = t + dt*(ω₁/ω₀)
  tᵢ   = tᵢ₋₁
  tᵢ₋₂ = t

  #stage 1
  @.. uᵢ₋₁ = uprev + (dt*ω₁/ω₀)*k

  for i in 2:mdeg
    Tᵢ = 2*ω₀*Tᵢ₋₁ - Tᵢ₋₂
    μ   = 2*ω₁*Tᵢ₋₁/Tᵢ
    ν   = 2*ω₀*Tᵢ₋₁/Tᵢ
    κ = - Tᵢ₋₂/Tᵢ
    integrator.f(k,uᵢ₋₁,p,tᵢ₋₁)
    @.. u = dt*μ*k + ν*uᵢ₋₁ + κ*uᵢ₋₂
    (i == mdeg - 1) && (integrator.g(gₘ₋₂,uᵢ₋₁,p,tᵢ₋₁); @.. u += α*W.dW*gₘ₋₂)
    (i == mdeg) && (integrator.g(gₘ₋₁,uᵢ₋₁,p,tᵢ₋₁); @.. u = β*W.dW*gₘ₋₂ + γ*W.dW*gₘ₋₁)
    if i < mdeg
      tᵢ = dt*μ + ν*tᵢ₋₁ + κ*tᵢ₋₂
      @.. uᵢ₋₂ = uᵢ₋₁
      @.. uᵢ₋₁ = u_i
      tᵢ₋₂ = tᵢ₋₁
      tᵢ₋₁ = tᵢ
      Tᵢ₋₂ = Tᵢ₋₁
      Tᵢ₋₁ = Tᵢ
    end
  end
  integrator.u = u
end