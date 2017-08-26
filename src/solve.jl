function solve(
  prob::AbstractRODEProblem,
alg::algType,timeseries=[],ts=[],ks=[],recompile::Type{Val{recompile_flag}}=Val{true};
kwargs...) where {algType<:Union{AbstractRODEAlgorithm,AbstractSDEAlgorithm},recompile_flag}

  integrator = init(prob,alg,timeseries,ts,ks,recompile;kwargs...)
  solve!(integrator)
  integrator.sol
end

function init(
  prob::AbstractRODEProblem{uType,tType,isinplace,ND},
  alg::algType,timeseries_init=uType[],ts_init=tType[],ks_init=[],
  recompile::Type{Val{recompile_flag}}=Val{true};
  dt = tType(0),
  timeseries_steps::Int = 1,
  save_noise = true,
  saveat = tType[],tstops = tType[],d_discontinuities= tType[],
  save_timeseries = nothing,
  save_everystep = isempty(saveat),
  save_idxs = nothing,
  save_start = true,
  dense = save_everystep,
  calck = (!isempty(setdiff(saveat,tstops)) || dense),
  adaptive=isadaptive(alg),gamma=9//10,
  abstol=1e-2,reltol=1e-2,
  qmax=qmax_default(alg),qmin=qmin_default(alg),
  failfactor = 2,
  qoldinit=1//10^4, fullnormalize=true,
  beta2=beta2_default(alg),
  beta1=beta1_default(alg,beta2),
  delta=1//6,maxiters = 1e9,
  dtmax=tType((prob.tspan[end]-prob.tspan[1])),
  dtmin=tType <: AbstractFloat ? tType(1000)*eps(tType) : tType(1//10^(10)),
  internalnorm=ODE_DEFAULT_NORM,
  unstable_check = ODE_DEFAULT_UNSTABLE_CHECK,
  isoutofdomain = ODE_DEFAULT_ISOUTOFDOMAIN,
  verbose = true,force_dtmin = false,
  advance_to_tstop = false,stop_at_next_tstop=false,
  progress_steps=1000,
  progress=false, progress_message = ODE_DEFAULT_PROG_MESSAGE,
  progress_name="SDE",
  userdata=nothing,callback=nothing,
  timeseries_errors = true, dense_errors=false,
  initialize_integrator=true,
  kwargs...) where {uType,tType,isinplace,algType<:Union{AbstractRODEAlgorithm,AbstractSDEAlgorithm},ND,recompile_flag}

  if save_timeseries != nothing
    warn("save_timeseries is deprecated. Use save_everystep instead")
    save_everystep = save_timeseries
  end

  if typeof(prob.f)<:Tuple
    if min((mm != I for mm in prob.mass_matrix)...)
      error("This solver is not able to use mass matrices.")
    end
  elseif prob.mass_matrix != I && !alg_mass_matrix_compatible(alg)
    error("This solver is not able to use mass matrices.")
  end

  if (typeof(prob.noise)<:NoiseProcess) && typeof(prob.noise.bridge)<:Void && adaptive
    error("Bridge function must be given for adaptivity. Either declare this function in noise process or set adaptive=false")
  end

  if !alg_compatible(prob,alg)
    error("The algorithm is not compatible with the chosen noise type. Please see the documentation on the solver methods")
  end

  noise = prob.noise
  tspan = prob.tspan
  tdir = sign(tspan[end]-tspan[1])

  T = tType(tspan[2])
  t = tType(tspan[1])

  if !adaptive && dt == tType(0) && isempty(tstops)
      error("Fixed timestep methods require a choice of dt or choosing the tstops")
  end

  if tspan[1] == tspan[2]
    error("Timespan is trivial")
  end

  d_discontinuities_col = collect(d_discontinuities)

  if tdir>0
    tstops_internal = binary_minheap(convert(Vector{tType},append!(collect(tstops),d_discontinuities_col)))
  else
    tstops_internal = binary_maxheap(convert(Vector{tType},append!(collect(tstops),d_discontinuities_col)))
  end

  if !isempty(tstops) && tstops[end] != tspan[2]
    push!(tstops_internal,tspan[2])
  elseif isempty(tstops)
    push!(tstops_internal,tspan[2])
  end

  if top(tstops_internal) == tspan[1]
    pop!(tstops_internal)
  end
  f = prob.f
  if typeof(prob) <: AbstractSDEProblem
    g = prob.g
  else
    g = nothing
  end
  u0 = prob.u0
  uEltype = recursive_eltype(u0)

  if typeof(prob.u0) <: Array
    u = recursivecopy(prob.u0)
  elseif typeof(prob.u0) <: Number
    u = prob.u0
  elseif typeof(prob.u0) <: Tuple
    u = ArrayPartition(prob.u0,Val{true})
  else
    u = deepcopy(prob.u0)
  end

  ks = Vector{uType}(0)

  order = alg_order(alg)

  if typeof(alg) <: SRI || typeof(alg) <: SRA
    order = alg.tableau.order
  end

  dtmax > 0 && tdir < 0 && (dtmax *= tdir) # Allow positive dtmax, but auto-convert
  # dtmin is all abs => does not care about sign already.
  if dt == zero(dt) && adaptive
    dt = tType(sde_determine_initdt(u,t,tdir,dtmax,abstol,reltol,internalnorm,prob,order))
    if sign(dt)!=tdir && dt!=tType(0) && !isnan(dt)
      error("Automatic dt setting has the wrong sign. Exiting. Please report this error.")
    end
    if isnan(dt)
      warn("Automatic dt set the starting dt as NaN, causing instability.")
    end
  elseif adaptive && dt > zero(dt) && tdir < 0
    dt *= tdir # Allow positive dt, but auto-convert
  end

  if typeof(u) <: AbstractArray
    rate_prototype = similar(u/zero(t),indices(u)) # rate doesn't need type info
  else
    rate_prototype = u/zero(t)
  end
  rateType = typeof(rate_prototype) ## Can be different if united

  if ND <: Void
    noise_rate_prototype = rate_prototype
  else
    if typeof(prob) <: AbstractSDEProblem
      noise_rate_prototype = prob.noise_rate_prototype
    else
      noise_rate_prototype = prob.rand_prototype
    end
  end

  if typeof(saveat) <: Number
    saveat_vec = convert(Vector{tType},tspan[1]+saveat:saveat:(tspan[end]-saveat))
    # Exclude the endpoint because of floating point issues
  else
    saveat_vec =  convert(Vector{tType},collect(saveat))
  end

  if !isempty(saveat_vec) && saveat_vec[end] == tspan[2]
    pop!(saveat_vec)
  end

  if tdir>0
    saveat_internal = binary_minheap(saveat_vec)
  else
    saveat_internal = binary_maxheap(saveat_vec)
  end

  if !isempty(saveat_internal) && top(saveat_internal) == tspan[1]
    pop!(saveat_internal)
  end

  d_discontinuities_vec =  convert(Vector{tType},d_discontinuities_col)

  if tdir>0
    d_discontinuities_internal = binary_minheap(d_discontinuities_vec)
  else
    d_discontinuities_internal = binary_maxheap(d_discontinuities_vec)
  end

  callbacks_internal = CallbackSet(callback,prob.callback)

  uEltypeNoUnits = typeof(recursive_one(u))
  tTypeNoUnits   = typeof(recursive_one(t))

  ### Algorithm-specific defaults ###
  ksEltype = Vector{rateType}

  # Have to convert incase passed in wrong.
  timeseries = convert(Vector{uType},timeseries_init)
  ts = convert(Vector{tType},ts_init)
  #ks = convert(Vector{ksEltype},ks_init)
  alg_choice = Int[]

  if save_start
    saveiter = 1
    copyat_or_push!(ts,1,t)
    if save_idxs == nothing
      copyat_or_push!(timeseries,1,u)
    else
      copyat_or_push!(timeseries,1,u[save_idxs],Val{false})
    end
    #copyat_or_push!(ks,1,[rate_prototype])
  else
    saveiter = 0
  end

  uEltype = recursive_eltype(u)

  opts = SDEOptions(maxiters,timeseries_steps,save_everystep,adaptive,map(uEltype,abstol),
    map(uEltypeNoUnits,reltol),tTypeNoUnits(gamma),tTypeNoUnits(qmax),tTypeNoUnits(qmin),
    tTypeNoUnits(failfactor),
    dtmax,dtmin,internalnorm,save_idxs,
    tstops_internal,saveat_internal,d_discontinuities_internal,
    userdata,
    progress,progress_steps,
    progress_name,progress_message,
    timeseries_errors,dense_errors,
    tTypeNoUnits(beta1),tTypeNoUnits(beta2),map(uEltypeNoUnits,delta),tTypeNoUnits(qoldinit),
    dense,save_start,save_noise,
    callbacks_internal,isoutofdomain,unstable_check,verbose,calck,force_dtmin,
    advance_to_tstop,stop_at_next_tstop)

  progress ? (prog = Juno.ProgressBar(name=progress_name)) : prog = nothing

  notsaveat_idxs = Int[1]

  k = ksEltype[]

  if uType <: Array
    uprev = copy(u)
  else
    uprev = deepcopy(u)
  end

  dtcache = tType(dt)
  dtchangeable = true

  ## Modify the first dt for tstops
  if !isempty(tstops_internal)
    if adaptive
      if tdir > 0
        dt = min(abs(dt),abs(top(tstops_internal)-t)) # step! to the end
      else
        dt = -min(abs(dt),abs(top(tstops_internal)-t))
      end
    elseif dt == zero(t) && dtchangeable # Use integrator.opts.tstops
      dt = tdir*abs(top(tstops_internal)-t)
    elseif dtchangeable # always try to step! with dtcache, but lower if a tstops
      dt = tdir*min(abs(dtcache),abs(top(tstops_internal)-t)) # step! to the end
    end
  end
  ### Needs to be done before first rand


  sqdt = tdir*sqrt(abs(dt))


  if !(uType <: AbstractArray)
    rand_prototype = zero(u/u) # Strip units and type info
    randType = typeof(rand_prototype)
  else
    randElType = typeof(recursive_one(u)) # Strip units and type info
    if ND <: Void # noise_dim isn't set, so it's diagonal
      rand_prototype = similar(Array{randElType},indices(u))
      fill!(rand_prototype,zero(randElType))
    elseif typeof(prob) <: AbstractSDEProblem
      rand_prototype = similar(Vector{randElType},size(noise_rate_prototype,2))
      fill!(rand_prototype,zero(randElType))
    else
      rand_prototype = prob.rand_prototype
    end
    randType = typeof(rand_prototype) # Strip units and type info
  end

  prob.seed == 0 ? seed = rand(UInt64) : seed = prob.seed

  if typeof(prob.noise) <: Void
    if isinplace
      isadaptive(alg) ? rswm = RSWM(adaptivealg=:RSwM3) : rswm = RSWM(adaptivealg=:RSwM1)
      if alg_needs_extra_process(alg)
        W = WienerProcess!(t,rand_prototype,rand_prototype,
                           save_everystep=save_everystep,
                           timeseries_steps=timeseries_steps,
                           rswm=rswm,
                           rng = Xorshifts.Xoroshiro128Plus(seed))
      else
        W = WienerProcess!(t,rand_prototype,
                           save_everystep=save_everystep,
                           timeseries_steps=timeseries_steps,
                           rswm=rswm,
                           rng = Xorshifts.Xoroshiro128Plus(seed))
      end
    else
      if alg_needs_extra_process(alg)
        W = WienerProcess(t,rand_prototype,rand_prototype,
                           save_everystep=save_everystep,
                           timeseries_steps=timeseries_steps,
                           rng = Xorshifts.Xoroshiro128Plus(seed))
      else
        W = WienerProcess(t,rand_prototype,
                           save_everystep=save_everystep,
                           timeseries_steps=timeseries_steps,
                           rng = Xorshifts.Xoroshiro128Plus(seed))
      end
    end
  else
    W = prob.noise
    if !(typeof(W) <: NoiseGrid) && W.t[end] != t
      error("Starting time in the noise process is not the starting time of the simulation. The noise process should be re-initialized for repeated use")
    end
  end

  EEst = tTypeNoUnits(1)
  q = tTypeNoUnits(1)
  just_hit_tstop = false
  isout = false
  accept_step = false
  u_modified = false
  last_stepfail = false
  force_stepfail = false
  tprev = t
  iter = 0
  success_iter = 0
  q11 = tTypeNoUnits(1)

  rateType = typeof(u/t) ## Can be different if united

  cache = alg_cache(alg,prob,u,W.dW,W.dZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,Val{isinplace})

  id = LinearInterpolationData(timeseries,ts)

  sol = build_solution(prob,alg,ts,timeseries,W=W,
                calculate_error = false,
                interp = id, dense = dense, seed = seed)

  if recompile_flag == true
    FType = typeof(f)
    GType = typeof(g)
    SolType = typeof(sol)
    cacheType = typeof(cache)
  else
    FType = Function
    GType = Function
    SolType = AbstractODESolution
    cacheType =  OrdinaryDiffEqCache
  end

  integrator =    SDEIntegrator{typeof(alg),uType,uEltype,tType,tTypeNoUnits,
                  uEltypeNoUnits,typeof(W),rateType,typeof(sol),typeof(cache),
                  typeof(prog),FType,GType,typeof(opts),typeof(noise)}(
                  f,g,noise,uprev,tprev,t,u,tType(dt),tType(dt),tType(dt),dtcache,T,tdir,
                  just_hit_tstop,isout,accept_step,last_stepfail,force_stepfail,
                  dtchangeable,u_modified,
                  saveiter,
                  alg,sol,
                  cache,sqdt,W,
                  opts,iter,success_iter,prog,EEst,q,
                  tTypeNoUnits(qoldinit),q11)

  integrator.W.dt = integrator.dt
  DiffEqNoiseProcess.setup_next_step!(integrator.W)

  if initialize_integrator
    initialize!(integrator,integrator.cache)
    initialize!(callbacks_internal,t,u,integrator)
  end
  integrator
end

function solve!(integrator::SDEIntegrator)
  @inbounds while !isempty(integrator.opts.tstops)
    while integrator.tdir*integrator.t < integrator.tdir*top(integrator.opts.tstops)
      loopheader!(integrator)
      @sde_exit_condtions
      perform_step!(integrator,integrator.cache)
      loopfooter!(integrator)
      if isempty(integrator.opts.tstops)
        break
      end
    end
    handle_tstop!(integrator)
  end
  postamble!(integrator)

  if typeof(integrator.sol.prob.f) <: Tuple
    f = integrator.sol.prob.f[1]
  else
    f = integrator.sol.prob.f
  end

  if has_analytic(f)
    calculate_solution_errors!(integrator.sol;timeseries_errors=integrator.opts.timeseries_errors,dense_errors=integrator.opts.dense_errors)
  end
  integrator.sol.retcode = :Success
  nothing
end
