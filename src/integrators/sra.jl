@inline function perform_step!(integrator,cache::SRA1ConstantCache,f=integrator.f)
  @unpack t,dt,uprev,u,ΔW,ΔZ = integrator
  gpdt = integrator.g(t+dt,uprev)
  chi2 = (ΔW + ΔZ/sqrt(3))/2 #I_(1,0)/h
  k₁ = dt*integrator.f(t,uprev)
  k₂ = dt*integrator.f(t+3dt/4,uprev+3k₁/4 + 3chi2*integrator.g(t+dt,uprev)/2)
  E₁ = k₁ + k₂
  E₂ = chi2.*(integrator.g(t,uprev)-gpdt) #Only for additive!

  if integrator.opts.adaptive
    u = uprev + k₁/3 + 2k₂/3 + E₂ + ΔW*gpdt
    integrator.EEst = abs((integrator.opts.delta*E₁+E₂)./(integrator.opts.abstol + max(abs(uprev),abs(u))*integrator.opts.reltol))
  else
    u = uprev + k₁/3 + 2k₂/3 + E₂ + ΔW*gpdt
  end
  @pack integrator = t,dt,u
end

@inline function perform_step!(integrator,cache::SRA1Cache,f=integrator.f)
  @unpack chi2,tmp1,E₁,E₂,gt,k₁,k₂,gpdt,EEsttmp = integrator.cache
  @unpack t,dt,uprev,u,ΔW,ΔZ = integrator
  integrator.g(t,uprev,gt)
  integrator.g(t+dt,uprev,gpdt)
  integrator.f(t,uprev,k₁); k₁*=dt
  for i in eachindex(u)
    chi2[i] = (ΔW[i] + ΔZ[i]/sqrt(3))/2 #I_(1,0)/h
    tmp1[i] = uprev[i]+3k₁[i]/4 + 3chi2[i]*gpdt[i]/2
  end

  integrator.f(t+3dt/4,tmp1,k₂); k₂*=dt

  for i in eachindex(u)
    E₁[i] = k₁[i] + k₂[i]
    E₂[i] = chi2[i]*(gt[i]-gpdt[i]) #Only for additive!
  end

  if integrator.opts.adaptive
    for i in eachindex(u)
      u[i] = uprev[i] + k₁[i]/3 + 2k₂[i]/3 + E₂[i] + ΔW[i]*gpdt[i]
    end
    for i in eachindex(u)
      EEsttmp[i] = (integrator.opts.delta*E₁[i]+E₂[i])/(integrator.opts.abstol + max(abs(uprev[i]),abs(u[i]))*integrator.opts.reltol)
    end
    integrator.EEst = integrator.opts.internalnorm(EEsttmp)
  else
    for i in eachindex(u)
      u[i] = uprev[i] + k₁[i]/3 + 2k₂[i]/3 + E₂[i] + ΔW[i]*gpdt[i]
    end
  end
  @pack integrator = t,dt,u
end

@inline function perform_step!(integrator,cache::SRACache,f=integrator.f)
  @unpack t,dt,uprev,u,ΔW,ΔZ = integrator
  @unpack H0,A0temp,B0temp,ftmp,gtmp,chi2,atemp,btemp,E₁,E₁temp,E₂,EEsttmp = integrator.cache
  @unpack c₀,c₁,A₀,B₀,α,β₁,β₂,stages = integrator.cache.tab
  for i in eachindex(u)
    chi2[i] = .5*(ΔW[i] + ΔZ[i]/sqrt(3)) #I_(1,0)/h
  end
  for i in 1:stages
    H0[i][:]=zero(eltype(integrator.u))
  end
  for i = 1:stages
    A0temp[:] = zero(eltype(integrator.u))
    B0temp[:] = zero(eltype(integrator.u))
    for j = 1:i-1
      integrator.f(t + c₀[j]*dt,H0[j],ftmp)
      integrator.g(t + c₁[j]*dt,H0[j],gtmp)
      for k in eachindex(u)
        A0temp[k] += A₀[i,j]*ftmp[k]
        B0temp[k] += B₀[i,j]*gtmp[k]
      end
    end
    for j in eachindex(u)
      H0[i][j] = uprev[j] + A0temp[j]*dt + B0temp[j]*chi2[j]
    end
  end
  atemp[:] = zero(eltype(integrator.u))
  btemp[:] = zero(eltype(integrator.u))
  E₂[:]    = zero(eltype(integrator.u))
  E₁temp[:]= zero(eltype(integrator.u))

  for i = 1:stages
    integrator.f(t+c₀[i]*dt,H0[i],ftmp)
    integrator.g(t+c₁[i]*dt,H0[i],gtmp)
    for j in eachindex(u)
      atemp[j] += α[i]*ftmp[j]
      btemp[j] += (β₁[i]*ΔW[j])*gtmp[j]
      E₂[j]    += (β₂[i]*chi2[j])*gtmp[j]
      E₁temp[j] += ftmp[j]
    end
  end
  for i in eachindex(u)
    E₁[i] = dt*E₁temp[i]
  end

  if integrator.opts.adaptive
    for i in eachindex(u)
      u[i] = uprev[i] + dt*atemp[i] + btemp[i] + E₂[i]
    end
    for i in eachindex(u)
      EEsttmp[i] = (integrator.opts.delta*E₁[i]+E₂[i])/(integrator.opts.abstol + max(abs(uprev[i]),abs(u[i]))*integrator.opts.reltol)
    end
    integrator.EEst = integrator.opts.internalnorm(EEsttmp)
  else
    for i in eachindex(u)
      u[i] = uprev[i] + dt*atemp[i] + btemp[i] + E₂[i]
    end
  end
  @pack integrator = t,dt,u
end

@inline function perform_step!(integrator,cache::SRAConstantCache,f=integrator.f)
  @unpack c₀,c₁,A₀,B₀,α,β₁,β₂,stages,H0 = integrator.cache
  @unpack t,dt,uprev,u,ΔW,ΔZ = integrator
  chi2 = .5*(ΔW + ΔZ/sqrt(3)) #I_(1,0)/h
  H0[:]=zeros(stages)
  for i = 1:stages
    A0temp = zero(u)
    B0temp = zero(u)
    for j = 1:i-1
      A0temp += A₀[i,j]*integrator.f(t + c₀[j]*dt,H0[j])
      B0temp += B₀[i,j]*integrator.g(t + c₁[j]*dt,H0[j]) #H0[..,i] argument ignored
    end
    H0[i] = uprev + A0temp*dt + B0temp.*chi2
  end

  atemp = zero(u)
  btemp = zero(u)
  E₂    = zero(u)
  E₁temp= zero(u)

  for i = 1:stages
    ftemp = integrator.f(t+c₀[i]*dt,H0[i])
    E₁temp += ftemp
    atemp += α[i]*ftemp
    btemp += (β₁[i]*ΔW ).*integrator.g(t+c₁[i]*dt,H0[i]) #H0[i] argument ignored
    E₂    += (β₂[i]*chi2).*integrator.g(t+c₁[i]*dt,H0[i]) #H0[i] argument ignored
  end

  if integrator.opts.adaptive
    E₁ = dt*E₁temp
    u = uprev + dt*atemp + btemp + E₂
    integrator.EEst = abs((integrator.opts.delta*E₁+E₂)./(integrator.opts.abstol + max(abs(uprev),abs(u))*integrator.opts.reltol))
  else
    u = uprev + dt*atemp + btemp + E₂
  end
  @pack integrator = t,dt,u
end