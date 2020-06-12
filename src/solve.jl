function DiffEqBase.__solve(prob::DiffEqBase.AbstractRODEProblem,
                            alg::Union{AbstractRODEAlgorithm,AbstractSDEAlgorithm},
                            timeseries=[],ts=[],ks=nothing, # needed for variable rate
                            recompile::Type{Val{recompile_flag}}=Val{true};
                            kwargs...) where recompile_flag
  integrator = DiffEqBase.__init(prob,alg,timeseries,ts,recompile;kwargs...)
  solve!(integrator)
  integrator.sol
end

# Make it easy to grab the RODEProblem/SDEProblem/DiscreteProblem from the keyword arguments
concrete_prob(prob) = prob
concrete_prob(prob::JumpProblem) = prob.prob

function DiffEqBase.__init(
  _prob::Union{DiffEqBase.AbstractRODEProblem,JumpProblem},
  alg::Union{AbstractRODEAlgorithm,AbstractSDEAlgorithm},timeseries_init=typeof(_prob.u0)[],
  ts_init=eltype(concrete_prob(_prob).tspan)[],
  ks_init=nothing,
  recompile::Type{Val{recompile_flag}}=Val{true};
  saveat = (),
  tstops = (),
  d_discontinuities= (),
  save_idxs = nothing,
  save_everystep = isempty(saveat),
  save_noise = save_everystep && (typeof(concrete_prob(_prob).f) <: Tuple ?
               DiffEqBase.has_analytic(concrete_prob(_prob).f[1]) : DiffEqBase.has_analytic(concrete_prob(_prob).f)),
  save_on = true,
  save_start = save_everystep || isempty(saveat) || typeof(saveat) <: Number ? true : concrete_prob(_prob).tspan[1] in saveat,
  save_end = save_everystep || isempty(saveat) || typeof(saveat) <: Number ? true : concrete_prob(_prob).tspan[2] in saveat,
  callback=nothing,
  dense = false, # save_everystep && isempty(saveat),
  calck = false, #(!isempty(setdiff(saveat,tstops)) || dense),
  dt = eltype(concrete_prob(_prob).tspan)(0),
  adaptive = isadaptive(_prob,alg),
  gamma= isadaptive(alg) ? 9//10 : 0,
  abstol=nothing,
  reltol=nothing,
  qmax=qmax_default(alg),qmin=qmin_default(alg),
  qoldinit= isadaptive(alg) ? 1//10^4 : 0, fullnormalize=true,
  failfactor = 2,
  beta2=beta2_default(alg),
  beta1=beta1_default(alg,beta2),
  delta=delta_default(alg),
  maxiters = adaptive ? 1000000 : typemax(Int),
  dtmax=eltype(concrete_prob(_prob).tspan)((concrete_prob(_prob).tspan[end]-concrete_prob(_prob).tspan[1])),
  dtmin = typeof(one(eltype(concrete_prob(_prob).tspan))) <: AbstractFloat ? eps(eltype(concrete_prob(_prob).tspan)) :
          typeof(one(eltype(concrete_prob(_prob).tspan))) <: Integer ? 0 :
          eltype(concrete_prob(_prob).tspan)(1//10^(10)),
  internalnorm = ODE_DEFAULT_NORM,
  isoutofdomain = ODE_DEFAULT_ISOUTOFDOMAIN,
  unstable_check = ODE_DEFAULT_UNSTABLE_CHECK,
  verbose = true,force_dtmin = false,
  timeseries_errors = true, dense_errors=false,
  advance_to_tstop = false,stop_at_next_tstop=false,
  initialize_save = true,
  progress=false,progress_steps=1000,progress_name="SDE",
  progress_message = ODE_DEFAULT_PROG_MESSAGE,
  userdata=nothing,
  initialize_integrator=true,
  seed = UInt64(0), alias_u0=false, alias_jumps = Threads.threadid()==1,
  kwargs...) where recompile_flag

  prob = concrete_prob(_prob)

  _seed = if iszero(seed)
    if (!(typeof(prob) <: DiffEqBase.AbstractRODEProblem) || iszero(prob.seed))
      seed_multiplier()*rand(UInt64)
    else
      prob.seed
    end
  else
    seed
  end

  if typeof(_prob) <: JumpProblem
    if alias_jumps
      _prob = DiffEqJump.resetted_jump_problem(_prob, _seed)
    elseif _seed !== 0  
      DiffEqJump.reset_jump_problem!(_prob, _seed)
    end
  end

  # Grab the deepcopied version for caching reasons.
  prob = concrete_prob(_prob)

  if typeof(prob.f)<:Tuple
    if any(mm != I for mm in prob.f.mass_matrix)
      error("This solver is not able to use mass matrices.")
    end
  elseif typeof(prob) <: DiffEqBase.AbstractRODEProblem && prob.f.mass_matrix != I && !alg_mass_matrix_compatible(alg)
    error("This solver is not able to use mass matrices.")
  end

  if !isempty(saveat) && dense
    @warn("Dense output is incompatible with saveat. Please use the SavingCallback from the Callback Library to mix the two behaviors.")
  end

  if typeof(prob) <: DiffEqBase.AbstractRODEProblem && typeof(prob.noise)<:NoiseProcess && prob.noise.bridge === nothing && adaptive
    error("Bridge function must be given for adaptivity. Either declare this function in noise process or set adaptive=false")
  end

  if !alg_compatible(_prob,alg)
    error("The algorithm is not compatible with the chosen noise type. Please see the documentation on the solver methods")
  end

  progress && @logmsg(-1,progress_name,_id=_id = :StochasticDiffEq,progress=0)

  tType = eltype(prob.tspan)
  noise = prob isa DiffEqBase.AbstractRODEProblem ? prob.noise : nothing
  tspan = prob.tspan
  tdir = sign(tspan[end]-tspan[1])

  t = tspan[1]

  if !adaptive && iszero(dt) && isempty(tstops)
      error("Fixed timestep methods require a choice of dt or choosing the tstops")
  end

  f = prob.f
  p = prob.p
  g = prob isa DiffEqBase.AbstractSDEProblem ? prob.g : nothing

  if typeof(prob.u0) <: Tuple
    u = ArrayPartition(prob.u0,Val{true})
  else
    if alias_u0
      u = prob.u0
    else
      u = recursivecopy(prob.u0)
    end
  end

  uType = typeof(u)
  uBottomEltype = recursive_bottom_eltype(u)
  uBottomEltypeNoUnits = recursive_unitless_bottom_eltype(u)

  ks = ()

  uEltypeNoUnits = recursive_unitless_eltype(u)
  tTypeNoUnits   = typeof(one(tType))

  if abstol === nothing
    if uBottomEltypeNoUnits === uBottomEltype && !(uBottomEltype <: Integer)
      abstol_internal = real(convert(uBottomEltype,oneunit(uBottomEltype)*1//10^2))
    elseif uBottomEltype <: Integer
      abstol_internal = real(oneunit(uBottomEltype)*1//10^2)
    else
      abstol_internal = 0
    end
  else
    abstol_internal = real.(abstol)
  end

  if reltol === nothing
    if uBottomEltypeNoUnits === uBottomEltype && !(uBottomEltype <: Integer)
      reltol_internal = real(convert(uBottomEltype,oneunit(uBottomEltype)*1//10^2))
    elseif uBottomEltype <: Integer
      reltol_internal = real(oneunit(uBottomEltype)*1//10^2)
    else
      reltol_internal = 0
    end
  else
    reltol_internal = real.(reltol)
  end

  dtmax > zero(dtmax) && tdir < 0 && (dtmax *= tdir) # Allow positive dtmax, but auto-convert
  # dtmin is all abs => does not care about sign already.

  if isinplace(prob) && typeof(u) <: AbstractArray && eltype(u) <: Number && uBottomEltypeNoUnits == uBottomEltype # Could this be more efficient for other arrays?
    if !(typeof(u) <: ArrayPartition)
      rate_prototype = recursivecopy(u)
    else
      rate_prototype = similar(u, typeof.(oneunit.(recursive_bottom_eltype.(u.x))./oneunit(tType))...)
    end
  else
    if uBottomEltypeNoUnits == uBottomEltype
      rate_prototype = u
    else # has units!
      rate_prototype = u/oneunit(tType)
    end
  end
  rateType = typeof(rate_prototype) ## Can be different if united

  if is_diagonal_noise(prob)
    noise_rate_prototype = rate_prototype
  elseif prob isa DiffEqBase.AbstractRODEProblem
    if prob isa DiffEqBase.AbstractSDEProblem
      noise_rate_prototype = copy(prob.noise_rate_prototype)
    else
      noise_rate_prototype = copy(prob.rand_prototype)
    end
  else
    noise_rate_prototype = nothing
  end

  tstops_internal, saveat_internal, d_discontinuities_internal =
    tstop_saveat_disc_handling(tstops, saveat, d_discontinuities, tspan)



  ### Algorithm-specific defaults ###
  # if save_idxs === nothing
  #   ksEltype = Vector{rateType}
  # else
  #   ks_prototype = rate_prototype[save_idxs]
  #   ksEltype = Vector{typeof(ks_prototype)}
  # end

  # Have to convert incase passed in wrong.
  if save_idxs === nothing
    timeseries = convert(Vector{uType},timeseries_init)
  else
    u_initial = u[save_idxs]
    timeseries = convert(Vector{typeof(u_initial)},timeseries_init)
  end
  ts = convert(Vector{tType},ts_init)

  alg_choice = typeof(alg) <: StochasticDiffEqCompositeAlgorithm ? Int[] : ()

  if !adaptive && save_everystep && tspan[2]-tspan[1] != Inf
    iszero(dt) ? steps = length(tstops) : steps = ceil(Int,internalnorm((tspan[2]-tspan[1])/dt,tspan[1]))
    sizehint!(timeseries,steps+1)
    sizehint!(ts,steps+1)
  elseif save_everystep
    sizehint!(timeseries,50)
    sizehint!(ts,50)
  elseif !isempty(saveat_internal)
    sizehint!(timeseries,length(saveat_internal)+1)
    sizehint!(ts,length(saveat_internal)+1)
  else
    sizehint!(timeseries,2)
    sizehint!(ts,2)
  end

  if save_start
    saveiter = 1 # Starts at 1 so first save is at 2
    copyat_or_push!(ts,1,t)
    if save_idxs === nothing
      copyat_or_push!(timeseries,1,u)
    else
      copyat_or_push!(timeseries,1,u_initial,Val{false})
    end
    if typeof(alg) <: StochasticDiffEqCompositeAlgorithm
      copyat_or_push!(alg_choice,1,1)
    end
  else
    saveiter = 0
  end

  QT = tTypeNoUnits <: Integer ? typeof(qmin) : tTypeNoUnits

  uprev = recursivecopy(u)

  if !(uType <: AbstractArray)
    rand_prototype = zero(u/u) # Strip units and type info
    randType = typeof(rand_prototype)
  else
    randElType = uBottomEltypeNoUnits # Strip units and type info
    if is_diagonal_noise(prob)
      if typeof(u) <: SArray
        rand_prototype = zero(u) # TODO: Array{randElType} for units
      else
        rand_prototype = (u .- u)./sqrt(oneunit(t))
      end
    elseif prob isa DiffEqBase.AbstractSDEProblem
      rand_prototype = false .* noise_rate_prototype[1,:]
    elseif prob isa DiffEqBase.AbstractRODEProblem
      rand_prototype = copy(prob.rand_prototype)
    else
      rand_prototype = nothing
    end
    randType = typeof(rand_prototype) # Strip units and type info
  end

  if typeof(_prob) <: JumpProblem
    callbacks_internal = CallbackSet(callback,_prob.jump_callback)
  else
    callbacks_internal = CallbackSet(callback)
  end

  max_len_cb = DiffEqBase.max_vector_callback_length(callbacks_internal)
  if max_len_cb isa DiffEqBase.VectorContinuousCallback
    callback_cache = DiffEqBase.CallbackCache(max_len_cb.len,uBottomEltype,uBottomEltype)
  else
    callback_cache = nothing
  end

  if typeof(prob) <: DiffEqBase.AbstractRODEProblem && prob.noise === nothing
    rswm = isadaptive(alg) ? RSWM(adaptivealg=:RSwM3) : RSWM(adaptivealg=:RSwM1)
    if isinplace(prob)
      _rng = typeof(prob.u0) <: Array{Float64} && seed !== nothing ?
                                            VectorizedRNG.local_rng() :
                                            Xorshifts.Xoroshiro128Plus(_seed)
      if alg_needs_extra_process(alg)
        W = constructor(t,rand_prototype,rand_prototype,
                           save_everystep=save_noise,
                           rng = _rng)
      else
        if alg_needs_extra_process(alg)
          W = SimpleWienerProcess!(t,rand_prototype,rand_prototype,
                             save_everystep=save_noise,
                             rng = Xorshifts.Xoroshiro128Plus(_seed))
        else
          W = SimpleWienerProcess!(t,rand_prototype,
                             save_everystep=save_noise,
                             rng = Xorshifts.Xoroshiro128Plus(_seed))
        end
      end
      =#
    else
      #if isadaptive(alg) || callback !== nothing
        if alg_needs_extra_process(alg)
          W = WienerProcess(t,rand_prototype,rand_prototype,
                             save_everystep=save_noise,
                             rng = Xorshifts.Xoroshiro128Plus(_seed))
        else
          W = WienerProcess(t,rand_prototype,
                             save_everystep=save_noise,
                             rng = Xorshifts.Xoroshiro128Plus(_seed))
        end
      #=
      else
        if alg_needs_extra_process(alg)
          W = SimpleWienerProcess(t,rand_prototype,rand_prototype,
                             save_everystep=save_noise,
                             rng = Xorshifts.Xoroshiro128Plus(_seed))
        else
          W = SimpleWienerProcess(t,rand_prototype,
                             save_everystep=save_noise,
                             rng = Xorshifts.Xoroshiro128Plus(_seed))
        end
      end
      =#
    end
  elseif typeof(prob) <: DiffEqBase.AbstractRODEProblem
    W = prob.noise
    if hasfield(typeof(W),:t) && W.reset
      if W.t[end] != t
        reinit!(W,t)
      end
      # Reseed
      if typeof(W) <: NoiseProcess && W.reseed
        Random.seed!(W.rng,_seed)
      end
    elseif hasfield(typeof(W),:t) && W.t[end] != t
      error("Starting time in the noise process is not the starting time of the simulation. The noise process should be re-initialized for repeated use")
    end
  else # Only a jump problem
    @assert typeof(_prob) <: JumpProblem
    W = nothing
  end

  if typeof(_prob) <: JumpProblem && _prob.regular_jump !== nothing

    if !isnothing(_prob.regular_jump.mark_dist) == nothing # https://github.com/JuliaDiffEq/DifferentialEquations.jl/issues/250
      error("Mark distributions are currently not supported in SimpleTauLeaping")
    end

    jump_prototype = zeros(_prob.regular_jump.numjumps)
    c = _prob.regular_jump.c

    if isinplace(_prob.regular_jump)
      rate_constants = zeros(_prob.regular_jump.numjumps)
      _prob.regular_jump.rate(rate_constants,u./u,prob.p,tspan[1])
      P = CompoundPoissonProcess!(_prob.regular_jump.rate,t,jump_prototype,
                                  computerates = !alg_control_rate(alg) || !adaptive,
                                  save_everystep=save_noise,
                                  rng = Xorshifts.Xoroshiro128Plus(_seed))
      alg_control_rate(alg) && adaptive && P.cache.rate(P.cache.currate,u,p,tspan[1])
    else
      rate_constants = _prob.regular_jump.rate(u./u,prob.p,tspan[1])
      P = CompoundPoissonProcess(_prob.regular_jump.rate,t,jump_prototype,
                                 save_everystep=save_noise,
                                 computerates = !alg_control_rate(alg) || !adaptive,
                                 rng = Xorshifts.Xoroshiro128Plus(_seed))
      alg_control_rate(alg) && adaptive && (P.cache.currate = P.cache.rate(u,p,tspan[1]))
    end

  else
    jump_prototype = nothing
    c = nothing
    P = nothing
    rate_constants = nothing
  end

  dW,dZ = isnothing(W) ? (nothing,nothing) : (W.dW,W.dZ)

  cache = alg_cache(alg,prob,u,dW,dZ,p,rate_prototype,noise_rate_prototype,jump_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,f,t,dt,Val{isinplace(_prob)})

  if typeof(_prob) <: JumpProblem && typeof(prob) <: DiscreteProblem && typeof(prob) <: Integer
    id = DiffEqBase.ConstantInterpolation(ts,timeseries)
  else
    id = LinearInterpolationData(timeseries,ts)
  end

  opts = SDEOptions(maxiters,save_everystep,
                    adaptive,abstol_internal,
                    reltol_internal,QT(gamma),
                    QT(qmax),QT(qmin),
                    QT(failfactor),
                    tType(dtmax),tType(dtmin),internalnorm,save_idxs,
                    tstops_internal,saveat_internal,
                    d_discontinuities_internal,
                    tstops,saveat,d_discontinuities,
                    userdata,
                    progress,progress_steps,
                    progress_name,progress_message,
                    timeseries_errors,dense_errors,
                    QT(beta1),QT(beta2),
                    convert.(uBottomEltypeNoUnits,delta),
                    QT(qoldinit),
                    dense,save_on,save_start,save_end,save_noise,
                    callbacks_internal,isoutofdomain,unstable_check,
                    verbose,calck,force_dtmin,
                    advance_to_tstop,stop_at_next_tstop)

  destats = DiffEqBase.DEStats(0)
  if typeof(alg) <: Union{StochasticDiffEqCompositeAlgorithm,
                          StochasticDiffEqRODECompositeAlgorithm}
    sol = DiffEqBase.build_solution(prob,alg,ts,timeseries,W=W,
                                    destats = destats,
                                    calculate_error = false, alg_choice=alg_choice,
                                    interp = id, dense = dense, seed = _seed)
  else
    sol = DiffEqBase.build_solution(prob,alg,ts,timeseries,W=W,
                                    destats = destats,
                                    calculate_error = false,
                                    interp = id, dense = dense, seed = _seed)
  end

  if recompile_flag == true
    FType = typeof(f)
    GType = typeof(g)
    CType = typeof(c)
    SolType = typeof(sol)
    cacheType = typeof(cache)
  else
    FType = Function
    GType = Function
    CType = Function
    SolType = DiffEqBase.AbstractRODESolution
    cacheType = StochasticDiffEqCache
  end

  tprev = t
  dtcache = tType(dt)
  iter = 0
  u_modified = false
  eigen_est = inv(one(tType)) # rate/state = (state/time)/state = 1/t units
  EEst = tTypeNoUnits(1)
  just_hit_tstop = false
  isout = false
  accept_step = false
  force_stepfail = false
  last_stepfail = false
  event_last_time = 0
  vector_event_last_time = 1
  last_event_error = zero(uBottomEltypeNoUnits)
  dtchangeable = true
  q11 = tTypeNoUnits(1)
  success_iter = 0
  q = tTypeNoUnits(1)

  integrator =    SDEIntegrator{typeof(alg),isinplace(prob),uType,
                  uBottomEltype,tType,typeof(tdir),typeof(p),
                  typeof(eigen_est),QT,
                  uEltypeNoUnits,typeof(W),typeof(P),rateType,typeof(sol),typeof(cache),
                  FType,GType,CType,typeof(opts),typeof(noise),typeof(last_event_error),typeof(callback_cache),typeof(rate_constants)}(
                  f,g,c,noise,uprev,tprev,t,u,p,tType(dt),tType(dt),tType(dt),dtcache,tspan[2],tdir,
                  just_hit_tstop,isout,event_last_time,vector_event_last_time,last_event_error,accept_step,
                  last_stepfail,force_stepfail,
                  dtchangeable,u_modified,
                  saveiter,
                  alg,sol,
                  cache,callback_cache,tType(dt),W,P,rate_constants,
                  opts,iter,success_iter,eigen_est,EEst,q,
                  QT(qoldinit),q11,destats)

  if initialize_integrator
    initialize_callbacks!(integrator, initialize_save)
    initialize!(integrator,integrator.cache)
    save_start && typeof(alg) <: Union{StochasticDiffEqCompositeAlgorithm,
                                       StochasticDiffEqRODECompositeAlgorithm} && copyat_or_push!(alg_choice,1,integrator.cache.current)
  end

  handle_dt!(integrator)

  ## Modify the first dt for tstops
  modify_dt_for_tstops!(integrator)
  ### Needs to be done before first rand
  integrator.sqdt = integrator.tdir*sqrt(abs(integrator.dt))

  !isnothing(integrator.W) && (integrator.W.dt = integrator.dt)
  !isnothing(integrator.P) && (integrator.P.dt = integrator.dt)

  DiffEqNoiseProcess.setup_next_step!(integrator::SDEIntegrator)

  integrator
end

function DiffEqBase.solve!(integrator::SDEIntegrator)
  @inbounds while !isempty(integrator.opts.tstops)
    while integrator.tdir*integrator.t < top(integrator.opts.tstops)
      loopheader!(integrator)
      if check_error!(integrator) != :Success
        return integrator.sol
      end
      perform_step!(integrator,integrator.cache)
      loopfooter!(integrator)
      if isempty(integrator.opts.tstops)
        break
      end
    end
    handle_tstop!(integrator)
  end
  postamble!(integrator)

  f = typeof(integrator.sol.prob.f) <: Tuple ? integrator.sol.prob.f[1] : integrator.sol.prob.f

  if DiffEqBase.has_analytic(f)
    DiffEqBase.calculate_solution_errors!(integrator.sol;timeseries_errors=integrator.opts.timeseries_errors,dense_errors=integrator.opts.dense_errors)
  end
  if integrator.sol.retcode != :Default
    return integrator.sol
  end
  integrator.sol = DiffEqBase.solution_new_retcode(integrator.sol,:Success)
end

# Helpers

function handle_dt!(integrator)
  if iszero(integrator.dt) && integrator.opts.adaptive
    auto_dt_reset!(integrator)
    if sign(integrator.dt)!=integrator.tdir && !iszero(integrator.dt) && !isnan(integrator.dt)
      error("Automatic dt setting has the wrong sign. Exiting. Please report this error.")
    end
    if isnan(integrator.dt)
      if integrator.opts.verbose
        @warn("Automatic dt set the starting dt as NaN, causing instability.")
      end
    end
  elseif integrator.opts.adaptive && integrator.dt > zero(integrator.dt) && integrator.tdir < 0
    integrator.dt *= integrator.tdir # Allow positive dt, but auto-convert
  end
end

function tstop_saveat_disc_handling(tstops, saveat, d_discontinuities, tspan)
  t0, tf = tspan
  tType = eltype(tspan)
  tdir = sign(tf - t0)

  tdir_t0 = tdir * t0
  tdir_tf = tdir * tf

  # time stops
  tstops_internal = BinaryMinHeap{tType}()
  if isempty(d_discontinuities) && isempty(tstops) # TODO: Specialize more
    push!(tstops_internal, tdir_tf)
  else
    for t in tstops
      tdir_t = tdir * t
      tdir_t0 < tdir_t ≤ tdir_tf && push!(tstops_internal, tdir_t)
    end

    for t in d_discontinuities
      tdir_t = tdir * t
      tdir_t0 < tdir_t ≤ tdir_tf && push!(tstops_internal, tdir_t)
    end

    push!(tstops_internal, tdir_tf)
  end

  # saving time points
  saveat_internal = BinaryMinHeap{tType}()
  if typeof(saveat) <: Number
    directional_saveat = tdir * abs(saveat)
    for t in (t0 + directional_saveat):directional_saveat:tf
      push!(saveat_internal, tdir * t)
    end
  elseif !isempty(saveat)
    for t in saveat
      tdir_t = tdir * t
      tdir_t0 < tdir_t < tdir_tf && push!(saveat_internal, tdir_t)
    end
  end

  # discontinuities
  d_discontinuities_internal = BinaryMinHeap{tType}()
  sizehint!(d_discontinuities_internal.valtree, length(d_discontinuities))
  for t in d_discontinuities
    push!(d_discontinuities_internal, tdir * t)
  end

  tstops_internal, saveat_internal, d_discontinuities_internal
end

function initialize_callbacks!(integrator, initialize_save = true)
  t = integrator.t
  u = integrator.u
  callbacks = integrator.opts.callback
  integrator.u_modified = true

  u_modified = initialize!(callbacks,u,t,integrator)

  # if the user modifies u, we need to fix previous values before initializing
  # FSAL in order for the starting derivatives to be correct
  if u_modified

    if isinplace(integrator.sol.prob)
      recursivecopy!(integrator.uprev,integrator.u)
    else
      integrator.uprev = integrator.u
    end

    if initialize_save &&
      (any((c)->c.save_positions[2],callbacks.discrete_callbacks) ||
      any((c)->c.save_positions[2],callbacks.continuous_callbacks))
      savevalues!(integrator,true)
    end
  end

  # reset this as it is now handled so the integrators should proceed as normal
  integrator.u_modified = false
end
