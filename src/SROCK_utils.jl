# This function calculates the largest eigenvalue
# (absolute value wise) by power iteration.
function maxeig!(integrator, cache::StochasticDiffEqConstantCache)
  isfirst = integrator.iter == 1 || integrator.u_modified
  @unpack t, dt, uprev, u, p = integrator
  maxiter =  50
  safe =  1.2
  fsalfirst = integrator.f(uprev, p, t)
  # Initial guess for eigenvector `z`
  if isfirst
    fz = fsalfirst
    z = integrator.f(fz, p, t)
  else
    z = cache.zprev
  end
  # Perturbation
  u_norm = integrator.opts.internalnorm(uprev,t)
  z_norm = integrator.opts.internalnorm(z,t)
  pert   = eps(u_norm)
  sqrt_pert = sqrt(pert)
  is_u_zero = u_norm == zero(u_norm)
  is_z_zero = z_norm == zero(z_norm)
  # Normalize `z` such that z-u lie in a circle
  if ( !is_u_zero && !is_z_zero )
    dz_u = u_norm * sqrt_pert
    quot = dz_u/z_norm
    z = uprev + quot*z
  elseif !is_u_zero
    dz_u = u_norm * sqrt_pert
    z = uprev + uprev*dz_u
  elseif !is_z_zero
    dz_u = pert
    quot = dz_u/z_norm
    z *= quot
  else
    dz_u = pert
    z = dz_u
  end # endif
  # Start power iteration
  integrator.eigen_est = 0
  for iter in 1:maxiter
    fz = integrator.f(z, p, t)
    tmp = fz - fsalfirst

    Δ  = integrator.opts.internalnorm(tmp,t)
    eig_prev = integrator.eigen_est
    integrator.eigen_est = Δ/dz_u * safe
    # Convergence
    if iter >= 2 && abs(eig_prev - integrator.eigen_est) < integrator.eigen_est*0.05
      # Store the eigenvector
      cache.zprev = z - uprev
      return true
    end

    # Next `z`
    if Δ != zero(Δ)
      quot = dz_u/Δ
      z = uprev + quot*tmp
    else
      # An arbitrary change on `z`
      nind = length(z)
      if (nind != 1)
        ind = 1 + iter % nind
        z[ind] = uprev[ind] - (z[ind] - uprev[ind])
      else
        z = -z
      end
    end
  end
  return false
end

function maxeig!(integrator, cache::StochasticDiffEqMutableCache)
  isfirst = integrator.iter == 1 || integrator.u_modified
  @unpack t, dt, uprev, u, p = integrator
  fz, z, atmp, fsalfirst = cache.k, cache.tmp, cache.atmp, cache.fsalfirst
  integrator.f(fsalfirst, uprev, p, t)
  ccache = cache.constantcache
  maxiter =  50
  safe = 1.2

  # Initial guess for eigenvector `z`
  if isfirst
    @.. fz = fsalfirst
    integrator.f(z, fz, p, t)
    integrator.destats.nf += 1
  else
    @.. z = ccache.zprev
  end
  # Perturbation
  u_norm = integrator.opts.internalnorm(uprev,t)
  z_norm = integrator.opts.internalnorm(z,t)
  pert   = eps(u_norm)
  sqrt_pert = sqrt(pert)
  is_u_zero = u_norm == zero(u_norm)
  is_z_zero = z_norm == zero(z_norm)
  # Normalize `z` such that z-u lie in a circle
  if ( !is_u_zero && !is_z_zero )
    dz_u = u_norm * sqrt_pert
    quot = dz_u/z_norm
    @.. z = uprev + quot*z
  elseif !is_u_zero
    dz_u = u_norm * sqrt_pert
    @.. z = uprev + uprev*dz_u
  elseif !is_z_zero
    dz_u = pert
    quot = dz_u/z_norm
    @.. z *= quot
  else
    dz_u = pert
    @.. z = dz_u
  end # endif
  # Start power iteration
  integrator.eigen_est = 0
  for iter in 1:maxiter
    integrator.f(fz, z, p, t)
    integrator.destats.nf += 1
    @.. atmp = fz - fsalfirst

    Δ  = integrator.opts.internalnorm(atmp,t)
    eig_prev = integrator.eigen_est
    integrator.eigen_est = Δ/dz_u * safe
    # Convergence
    if iter >= 2 && abs(eig_prev - integrator.eigen_est) < integrator.eigen_est*0.05
      # Store the eigenvector
      @.. ccache.zprev = z - uprev
      return true
    end
    # Next `z`
    if Δ != zero(Δ)
      quot = dz_u/Δ
      @.. z = uprev + quot*atmp
    else
      # An arbitrary change on `z`
      nind = length(uprev)
      if (nind != 1)
        ind = 1 + iter % nind
        z[ind] = uprev[ind] - (z[ind] - uprev[ind])
      else
        z = -z
      end
    end
  end
  return false
end


# """
#     choosedeg!(cache) -> nothing
#
# Calculate `mdeg = ms[deg_index]` (the degree of the Chebyshev polynomial)
# and `cache.start` (the start index of recurrence parameters for that
# degree), where `recf` are the `μ,κ` pairs
# for the `mdeg` degree method. The `κ` for `stage-1` for every degree
# is 0 therefore it's not included in `recf`
#   """
# function choosedeg!(cache::T) where T
#   isconst = T <: OrdinaryDiffEqConstantCache
#   isconst || ( cache = cache.constantcache )
#   start = 1
#   @inbounds for i in 1:size(cache.ms,1)
#     if cache.ms[i] >= cache.mdeg
#       cache.deg_index = i
#       cache.mdeg = cache.ms[i]
#       cache.start = start
#       break
#     end
#     start += cache.ms[i]*2 - 1
#   end
#   return nothing
# end
#
#
# function choosedeg_SERK!(integrator,cache::T) where T
#   isconst = T <: OrdinaryDiffEqConstantCache
#   isconst || ( cache = cache.constantcache )
#   @unpack ms = cache
#   start = 1
#   @inbounds for i in 1:size(ms,1)
#     if ms[i] < cache.mdeg
#       start += ms[i]+1
#     else
#       cache.start = start
#       cache.mdeg = ms[i]
#       break
#     end
#   end
#   if integrator.alg isa ESERK5
#     if cache.mdeg <= 20
#       cache.internal_deg = 2
#     elseif cache.mdeg <= 50
#       cache.internal_deg = 5
#     elseif cache.mdeg <= 100
#       cache.internal_deg = 10
#     elseif cache.mdeg <= 500
#       cache.internal_deg = 50
#     elseif cache.mdeg <= 1000
#       cache.internal_deg = 100
#     elseif cache.mdeg <= 2000
#       cache.internal_deg = 200
#     end
#   end
#
#   if integrator.alg isa SERK2v2
#     cache.internal_deg = cache.mdeg/10
#   end
#   return nothing
# end