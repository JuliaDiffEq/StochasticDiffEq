@compat abstract type StochasticDiffEqCache <: DECache end
@compat abstract type StochasticDiffEqConstantCache <: StochasticDiffEqCache end
@compat abstract type StochasticDiffEqMutableCache <: StochasticDiffEqCache end

type StochasticCompositeCache{T,F} <: StochasticDiffEqCache
  caches::T
  choice_function::F
  current::Int
end

function alg_cache{T,algType<:StochasticCompositeAlgorithm}(alg::algType,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{T}})
  caches = map((x)->alg_cache(x,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,Val{T}),alg.algs)
  StochasticCompositeCache(caches,alg.choice_function,1)
end

immutable EMConstantCache <: StochasticDiffEqConstantCache end
immutable EMCache{uType,rateType,rateNoiseType,rateNoiseCollectionType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  tmp::uType
  rtmp1::rateType
  rtmp2::rateNoiseType
  rtmp3::rateNoiseCollectionType
end

u_cache(c::EMCache) = ()
du_cache(c::EMCache) = (c.rtmp1,c.rtmp2)

alg_cache(alg::EM,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}}) = EMConstantCache()

function alg_cache(alg::EM,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})
  tmp = similar(u); rtmp1 = zeros(rate_prototype);
  rtmp2 = zeros(noise_rate_prototype)
  if is_diagonal_noise(prob)
    rtmp3 = rtmp2
  else
    rtmp3 = zeros(rate_prototype)
  end
  EMCache(u,uprev,tmp,rtmp1,rtmp2,rtmp3)
end

immutable SplitEMConstantCache <: StochasticDiffEqConstantCache end
immutable SplitEMCache{uType,rateType,rateNoiseType,rateNoiseCollectionType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  tmp::uType
  rtmp1::rateType
  rtmp2::rateNoiseType
  rtmp3::rateNoiseCollectionType
end

u_cache(c::SplitEMCache) = ()
du_cache(c::SplitEMCache) = (c.rtmp1,c.rtmp2)

alg_cache(alg::SplitEM,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}}) = SplitEMConstantCache()

function alg_cache(alg::SplitEM,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})
  tmp = similar(u); rtmp1 = zeros(rate_prototype);
  rtmp2 = zeros(noise_rate_prototype)
  if is_diagonal_noise(prob)
    rtmp3 = rtmp2
  else
    rtmp3 = zeros(rate_prototype)
  end
  SplitEMCache(u,uprev,tmp,rtmp1,rtmp2,rtmp3)
end

immutable IIF1MConstantCache{vecuType,rhsType,nl_rhsType} <: StochasticDiffEqConstantCache
  uhold::vecuType
  rhs::rhsType
  nl_rhs::nl_rhsType
end

immutable IIF1MCache{uType,vecuType,DiffCacheType,rhsType,nl_rhsType,rateType,rateNoiseType,rateNoiseCollectionType,NoiseTmpType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  uhold::vecuType
  dual_cache::DiffCacheType
  tmp::uType
  rhs::rhsType
  nl_rhs::nl_rhsType
  rtmp1::rateType
  rtmp2::rateNoiseType
  rtmp3::rateNoiseCollectionType
  noise_tmp::NoiseTmpType
end

u_cache(c::IIF1MCache)    = (c.uprev2,c.u_old)
du_cache(c::IIF1MCache)   = (c.rtmp1,c.rtmp2,c.rtmp3,c.tmp)
vecu_cache(c::IIF1MCache) = (c.uhold,)
dual_cache(c::IIF1MCache) = (c.dual_cache,)

function alg_cache(alg::IIF1M,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}})
  uhold = Vector{typeof(u)}(1)
  tmp = zero(rate_prototype)
  rhs = RHS_IIF1M_Scalar(f,tmp,t,t)
  nl_rhs = alg.nlsolve(Val{:init},rhs,uhold)
  IIF1MConstantCache(uhold,rhs,nl_rhs)
end

function alg_cache(alg::IIF1M,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})

  tmp = similar(u,indices(u)); rtmp1 = zeros(rate_prototype)
  dual_cache = DiffCache(u,Val{DiffEqDiffTools.determine_chunksize(u,DiffEqDiffTools.get_chunksize(alg.nlsolve))})
  uhold = vec(u) # this makes uhold the same values as integrator.u
  rhs = RHS_IIF1(f,tmp,t,t,dual_cache,size(u),eachindex(u))
  nl_rhs = alg.nlsolve(Val{:init},rhs,uhold)
  noise_tmp = tmp

  rtmp2 = zeros(noise_rate_prototype)
  if is_diagonal_noise(prob)
    rtmp3 = rtmp2
  else
    rtmp3 = zeros(rate_prototype)
  end
  IIF1MCache(u,uprev,uhold,dual_cache,tmp,rhs,nl_rhs,rtmp1,rtmp2,rtmp3,noise_tmp)
end

immutable IIF1MilConstantCache{vecuType,rhsType,nl_rhsType} <: StochasticDiffEqConstantCache
  uhold::vecuType
  rhs::rhsType
  nl_rhs::nl_rhsType
end
immutable IIF1MilCache{uType,vecuType,DiffCacheType,rhsType,nl_rhsType,rateType,rateNoiseType,rateNoiseCollectionType,NoiseTmpType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  uhold::vecuType
  dual_cache::DiffCacheType
  tmp::uType
  rhs::rhsType
  nl_rhs::nl_rhsType
  rtmp1::rateType
  rtmp2::rateNoiseType
  rtmp3::rateNoiseCollectionType
  noise_tmp::NoiseTmpType
end

u_cache(c::IIF1MilCache)    = (c.uprev2,c.u_old)
du_cache(c::IIF1MilCache)   = (c.rtmp1,c.rtmp2,c.rtmp3,c.tmp)
vecu_cache(c::IIF1MilCache) = (c.uhold,)
dual_cache(c::IIF1MilCache) = (c.dual_cache,)

function alg_cache(alg::IIF1Mil,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}})
  uhold = Vector{typeof(u)}(1)
  C = zero(rate_prototype)
  rhs = RHS_IIF1M_Scalar(f,C,t,t)
  nl_rhs = alg.nlsolve(Val{:init},rhs,uhold)
  IIF1MilConstantCache(uhold,rhs,nl_rhs)
end

function alg_cache(alg::IIF1Mil,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})
  tmp = similar(u,indices(u)); rtmp1 = zeros(rate_prototype)
  dual_cache = DiffCache(u,Val{DiffEqDiffTools.determine_chunksize(u,DiffEqDiffTools.get_chunksize(alg.nlsolve))})
  uhold = vec(u) # this makes uhold the same values as integrator.u
  rhs = RHS_IIF1(f,tmp,t,t,dual_cache,size(u),eachindex(u))
  nl_rhs = alg.nlsolve(Val{:init},rhs,uhold)
  noise_tmp = similar(noise_rate_prototype)

  rtmp2 = zeros(noise_rate_prototype)
  if is_diagonal_noise(prob)
    rtmp3 = rtmp2
  else
    rtmp3 = zeros(rate_prototype)
  end
  IIF1MilCache(u,uprev,uhold,dual_cache,tmp,rhs,nl_rhs,rtmp1,rtmp2,rtmp3,noise_tmp)
end


immutable EulerHeunConstantCache <: StochasticDiffEqConstantCache end
immutable EulerHeunCache{uType,rateType,rateNoiseType,rateNoiseCollectionType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  tmp::uType
  ftmp1::rateType
  ftmp2::rateType
  nrtmp::rateNoiseCollectionType
  gtmp1::rateNoiseType
  gtmp2::rateNoiseType
end

u_cache(c::EulerHeunCache) = ()
du_cache(c::EulerHeunCache) = (c.rtmp1,c.rtmp2,c.rtmp3,c.rtmp4)

alg_cache(alg::EulerHeun,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}}) = EulerHeunConstantCache()

function alg_cache(alg::EulerHeun,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})
  tmp = similar(u); ftmp1 = zeros(rate_prototype); ftmp2 = zeros(rate_prototype)
  nrtmp = zeros(rate_prototype)
  gtmp1 = zeros(noise_rate_prototype); gtmp2 = zeros(noise_rate_prototype)
  EulerHeunCache(u,uprev,tmp,ftmp1,ftmp2,nrtmp,gtmp1,gtmp2)
end

immutable RandomEMConstantCache <: StochasticDiffEqConstantCache end
immutable RandomEMCache{uType,rateType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  tmp::uType
  rtmp::rateType
end

u_cache(c::RandomEMCache) = ()
du_cache(c::RandomEMCache) = (c.rtmp1,c.rtmp2)

alg_cache(alg::RandomEM,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}}) = RandomEMConstantCache()

function alg_cache(alg::RandomEM,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})
  tmp = similar(u); rtmp = zeros(rate_prototype)
  RandomEMCache(u,uprev,tmp,rtmp)
end

immutable RKMilConstantCache <: StochasticDiffEqConstantCache end
immutable RKMilCache{uType,rateType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  du1::rateType
  du2::rateType
  K::rateType
  tmp::uType
  L::rateType
end

u_cache(c::RKMilCache) = ()
du_cache(c::RKMilCache) = (c.du1,c.du2,c.K,c.L)

alg_cache(alg::RKMil,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}}) = RKMilConstantCache()

function alg_cache(alg::RKMil,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})
  du1 = zeros(rate_prototype); du2 = zeros(rate_prototype)
  K = zeros(rate_prototype); tmp = similar(u); L = zeros(rate_prototype)
  RKMilCache(u,uprev,du1,du2,K,tmp,L)
end

immutable SRA1ConstantCache <: StochasticDiffEqConstantCache end
alg_cache(alg::SRA1,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}}) = SRA1ConstantCache()

immutable SRA1Cache{randType,rateType,uType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  chi2::randType
  tmp1::uType
  E₁::rateType
  E₂::rateType
  gt::rateType
  k₁::rateType
  k₂::rateType
  gpdt::rateType
  tmp::uType
end

u_cache(c::SRA1Cache) = ()
du_cache(c::SRA1Cache) = (c.chi2,c.E₁,c.E₂,c.gt,c.k₁,c.k₂,c.gpdt)
user_cache(c::SRA1Cache) = (c.u,c.uprev,c.tmp,c.tmp1)

function alg_cache(alg::SRA1,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})
  chi2 = similar(ΔW)
  tmp1 = zeros(u)
  E₁ = zeros(rate_prototype); gt = zeros(rate_prototype); gpdt = zeros(rate_prototype)
  E₂ = zeros(rate_prototype); k₁ = zeros(rate_prototype); k₂ = zeros(rate_prototype)
  tmp = zeros(u)
  SRA1Cache(u,uprev,chi2,tmp1,E₁,E₂,gt,k₁,k₂,gpdt,tmp)
end

immutable SRAConstantCache{VType1,VType2,MType,uType} <: StochasticDiffEqConstantCache
  c₀::VType1
  c₁::VType1
  A₀::MType
  B₀::MType
  α::VType2
  β₁::VType2
  β₂::VType2
  stages::Int
  H0::Vector{uType}
end

function SRAConstantCache(tableau,rate_prototype)
  @unpack c₀,c₁,A₀,B₀,α,β₁,β₂ = tableau
  stages = length(α)
  H0 = Vector{typeof(rate_prototype)}(stages)
  SRAConstantCache(c₀,c₁,A₀',B₀',α,β₁,β₂,stages,H0)
end

function alg_cache(alg::SRA,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}})
  SRAConstantCache(alg.tableau,rate_prototype)
end

immutable SRACache{uType,rateType,tabType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  H0::Vector{uType}
  A0temp::rateType
  B0temp::rateType
  ftmp::rateType
  gtmp::rateType
  chi2::rateType
  atemp::rateType
  btemp::rateType
  E₁::rateType
  E₁temp::rateType
  E₂::rateType
  tmp::uType
  tab::tabType
end

u_cache(c::SRACache) = ()
du_cache(c::SRACache) = (c.A0temp,c.B0temp,c.ftmp,c.gtmp,c.chi2,c.chi2,c.atemp,
                          c.btemp,c.E₁,c.E₁temp,c.E₂)
user_cache(c::SRACache) = (c.u,c.uprev,c.tmp,c.H0...)

function alg_cache(alg::SRA,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})
  H0 = Vector{typeof(u)}(0)
  tab = SRAConstantCache(alg.tableau,rate_prototype)
  for i = 1:tab.stages
    push!(H0,zeros(u))
  end
  A0temp = zeros(rate_prototype); B0temp = zeros(rate_prototype)
  ftmp = zeros(rate_prototype); gtmp = zeros(rate_prototype); chi2 = zeros(rate_prototype)
  atemp = zeros(rate_prototype); btemp = zeros(rate_prototype); E₂ = zeros(rate_prototype); E₁temp = zeros(rate_prototype)
  E₁ = zeros(rate_prototype)
  tmp = zeros(u)
  SRACache(u,uprev,H0,A0temp,B0temp,ftmp,gtmp,chi2,atemp,btemp,E₁,E₁temp,E₂,tmp,tab)
end

immutable SRIConstantCache{VType1,VType2,MType,uType} <: StochasticDiffEqConstantCache
  c₀::VType1
  c₁::VType1
  A₀::MType
  A₁::MType
  B₀::MType
  B₁::MType
  α::VType2
  β₁::VType2
  β₂::VType2
  β₃::VType2
  β₄::VType2
  stages::Int
  H0::Vector{uType}
  H1::Vector{uType}
  error_terms::Int
end

function SRIConstantCache(tableau,rate_prototype,error_terms)
  @unpack c₀,c₁,A₀,A₁,B₀,B₁,α,β₁,β₂,β₃,β₄ = tableau
  stages = length(α)
  H0 = Array{typeof(rate_prototype)}(stages)
  H1 = Array{typeof(rate_prototype)}(stages)
  SRIConstantCache(c₀,c₁,A₀',A₁',B₀',B₁',α,β₁,β₂,β₃,β₄,stages,H0,H1,error_terms)
end

function alg_cache(alg::SRI,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}})
  SRIConstantCache(alg.tableau,rate_prototype,alg.error_terms)
end

immutable SRICache{randType,uType,rateType,tabType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  H0::Vector{uType}
  H1::Vector{uType}
  A0temp::rateType
  A1temp::rateType
  B0temp::rateType
  B1temp::rateType
  A0temp2::rateType
  A1temp2::rateType
  B0temp2::rateType
  B1temp2::rateType
  atemp::rateType
  btemp::rateType
  E₁::rateType
  E₂::rateType
  E₁temp::rateType
  ftemp::rateType
  gtemp::rateType
  chi1::randType
  chi2::randType
  chi3::randType
  tmp::uType
  tab::tabType
end

u_cache(c::SRICache) = ()
du_cache(c::SRICache) = (c.A0temp,c.A1temp,c.B0temp,c.B1temp,c.A0temp2,c.A1temp2,
                          c.B0temp2,c.B1temp2,c.atemp,c.btemp,c.E₁,c.E₂,c.E₁temp,
                          c.ftemp,c.gtemp,c.chi1,c.chi2,c.chi3)
user_cache(c::SRICache) = (c.u,c.uprev,c.tmp,c.H0...,c.H1...)

function alg_cache(alg::SRI,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})
  H0 = Vector{typeof(u)}(0)
  H1 = Vector{typeof(u)}(0)
  tab = SRIConstantCache(alg.tableau,rate_prototype,alg.error_terms)
  for i = 1:tab.stages
    push!(H0,zeros(u))
    push!(H1,zeros(u))
  end
  #TODO Reduce memory
  A0temp = zeros(rate_prototype); A1temp = zeros(rate_prototype)
  B0temp = zeros(rate_prototype); B1temp = zeros(rate_prototype)
  A0temp2 = zeros(rate_prototype); A1temp2 = zeros(rate_prototype)
  B0temp2 = zeros(rate_prototype); B1temp2 = zeros(rate_prototype)
  atemp = zeros(rate_prototype); btemp = zeros(rate_prototype)
  E₁ = zeros(rate_prototype); E₂ = zeros(rate_prototype); E₁temp = zeros(rate_prototype)
  ftemp = zeros(rate_prototype); gtemp = zeros(rate_prototype)
  chi1 = similar(ΔW); chi2 = similar(ΔW); chi3 = similar(ΔW)
  tmp = zeros(u)
  SRICache(u,uprev,H0,H1,A0temp,A1temp,B0temp,B1temp,A0temp2,A1temp2,B0temp2,B1temp2,
    atemp,btemp,E₁,E₂,E₁temp,ftemp,gtemp,chi1,chi2,chi3,tmp,tab)
end

immutable SRIW1ConstantCache <: StochasticDiffEqConstantCache end
alg_cache(alg::SRIW1,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{false}}) = SRIW1ConstantCache()

immutable SRIW1Cache{randType,uType,rateType} <: StochasticDiffEqMutableCache
  u::uType
  uprev::uType
  chi1::randType
  chi2::randType
  chi3::randType
  fH01o4::rateType
  g₁o2::rateType
  H0::uType
  H11::uType
  H12::uType
  H13::uType
  g₂o3::rateType
  Fg₂o3::rateType
  g₃o3::rateType
  Tg₃o3::rateType
  mg₁::rateType
  E₁::rateType
  E₂::rateType
  fH01::rateType
  fH02::rateType
  g₁::rateType
  g₂::rateType
  g₃::rateType
  g₄::rateType
  tmp::uType
end

u_cache(c::SRIW1Cache) = ()
du_cache(c::SRIW1Cache) = (c.chi1,c.chi2,c.chi3,c.fH01o4,c.g₁o2,c.g₂o3,c.Fg₂o3,c.g₃o3,c.Tg₃o3,c.mg₁,
                          c.E₁,c.E₂,c.fH01,c.fH02,c.g₁,c.g₂,c.g₃,c.g₄)
user_cache(c::SRIW1Cache) = (c.u,c.uprev,c.tmp,c.H0,c.H11,c.H12,c.H13)

function alg_cache(alg::SRIW1,prob,u,ΔW,ΔZ,rate_prototype,noise_rate_prototype,uEltypeNoUnits,tTypeNoUnits,uprev,f,t,::Type{Val{true}})
  chi1 = similar(ΔW)
  chi2 = similar(ΔW)
  chi3 = similar(ΔW)
  fH01o4 = zeros(rate_prototype)
  g₁o2 = zeros(rate_prototype)
  H0 = zeros(u)
  H11 = zeros(u)
  H12 = zeros(u)
  H13 = zeros(u)
  g₂o3 = zeros(rate_prototype)
  Fg₂o3 = zeros(rate_prototype)
  g₃o3 = zeros(rate_prototype)
  Tg₃o3 = zeros(rate_prototype)
  mg₁ = zeros(rate_prototype)
  E₁ = zeros(rate_prototype)
  E₂ = zeros(rate_prototype)
  fH01 = zeros(rate_prototype); fH02 = zeros(rate_prototype)
  g₁ = zeros(rate_prototype); g₂ = zeros(rate_prototype); g₃ = zeros(rate_prototype); g₄ = zeros(rate_prototype)
  tmp = zeros(u)
  SRIW1Cache(u,uprev,chi1,chi2,chi3,fH01o4,g₁o2,H0,H11,H12,H13,g₂o3,Fg₂o3,g₃o3,Tg₃o3,mg₁,E₁,E₂,fH01,fH02,g₁,g₂,g₃,g₄,tmp)
end
