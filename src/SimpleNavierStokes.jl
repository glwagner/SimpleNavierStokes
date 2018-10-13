module SimpleNavierStokes

export
  Problem,
  Setup,
  Grid,
  Vars,

  stepforward!,
  randomIC!

using
  FFTW,
  Random,
  Parameters

using LinearAlgebra: mul!, ldiv!

const dim = 2

"""
    @createarrays T dims a b c...

Create arrays of all zeros with element type `T`, size `dims`, and global names
`a`, `b`, `c` (for example). An arbitrary number of arrays may be created.
"""
macro createarrays(T, dims, vars...)
  expr = Expr(:block)
  append!(expr.args, [:($(esc(var)) = zeros($(esc(T)), $(esc(dims))); ) for var in vars])
  expr
end

struct Setup{T}
  nx::Int
  Lx::T
  nu::T
  dt::T
end

Setup(; T=Float64, nx=128, Lx=2π, nu=8e-5, dt=1e-2) = Setup{T}(nx, Lx, nu, dt)

struct Grid{T}
  nx::Int
  dx::T
  Lx::T
  x::Array{T,dim}
  y::Array{T,dim}
  k::Array{T,dim}
  l::Array{T,dim}
  Ksq::Array{T,dim}
  invKsq::Array{T,dim}
  fftplan::FFTW.rFFTWPlan{T,-1,false,2}
end

struct Vars{T}
  u::Array{T,dim}
  v::Array{T,dim}
  q::Array{T,dim}
  uq::Array{T,dim}
  vq::Array{T,dim}
  qh::Array{Complex{T},dim}
  qsh::Array{Complex{T},dim}
  psih::Array{Complex{T},dim}
  uh::Array{Complex{T},dim}
  vh::Array{Complex{T},dim}
  uqh::Array{Complex{T},dim}
  vqh::Array{Complex{T},dim}
  rhs::Array{Complex{T},dim}
end

function Grid(nx, Lx; T=typeof(Lx), effort=FFTW.MEASURE, nthreads=Sys.CPU_THREADS)
  # Construct the grid
  ny, Ly = nx, Lx
  dx, dy = Lx/nx, Ly/ny
  nk, nl = Int(nx/2+1), ny

  x = Array(reshape(0:dx:Lx-dx, (nx, 1)))
  y = Array(reshape(0:dy:Ly-dy, (1, ny)))
  k = Array(reshape(2π/Lx*(0:nk-1), (nk, 1)))
  l = Array(reshape(2π/Ly*cat(0:nl/2, -nl/2+1:-1, dims=1), (1, nl)))

  # Need K^2 = k^2 + l^2 and 1/K^2.
  Ksq = @. k^2 + l^2
  invKsq = @. 1/Ksq
  invKsq[1, 1] = 0 # eliminates 0th mode during inversion

  FFTW.set_num_threads(nthreads)
  fftplan = plan_rfft(Array{T,2}(undef, nx, ny); flags=effort)

  Grid{T}(nx, dx, Lx, x, y, k, l, Ksq, invKsq, fftplan)
end

Grid(s::Setup) = Grid(s.nx, s.Lx; T=typeof(s.Lx))

function Vars(T, nx, ny)
  nk, nl = Int(nx/2+1), ny
  @createarrays T (nx, ny) u v q uq vq
  @createarrays Complex{T} (nk, nl) qh qsh psih rhs uh vh uqh vqh
  Vars{T}(u, v, q, uq, vq, qh, qsh, psih, uh, vh, uqh, vqh, rhs)
end

Vars(g::Grid) = Vars(typeof(g.Lx), g.nx, g.ny)
Vars(s::Setup) = Vars(typeof(s.Lx), s.nx, s.nx)

mutable struct Problem{T}
  grid::Grid{T}
  vars::Vars{T}
  nu::T
  dt::T
  t::T
  step::Int
end

Problem(s::Setup) = Problem{typeof(s.Lx)}(Grid(s), Vars(s), s.nu, s.dt, 0, 0)

function calcrhs!(rhs, v, g, nu, dt)
  # Calculate right hand side of vorticity equation.
  v.qsh .= v.qh  # Necessary because irfft destroys its input.
  ldiv!(v.q, g.fftplan, v.qsh)

  @. v.uh =  im * g.l * g.invKsq * v.qh
  @. v.vh = -im * g.k * g.invKsq * v.qh

  ldiv!(v.u, g.fftplan, v.uh)
  ldiv!(v.v, g.fftplan, v.vh)

  @. v.uq = v.u*v.q
  @. v.vq = v.v*v.q

  mul!(v.uqh, g.fftplan, v.uq)
  mul!(v.vqh, g.fftplan, v.vq)

  @. rhs = -im*g.k*v.uqh - im*g.l*v.vqh - nu*g.Ksq*v.qh
  nothing
end

function stepforward!(prob)
  calcrhs!(prob.vars.rhs, prob.vars, prob.grid, prob.nu, prob.dt)
  @. prob.vars.qh += prob.dt*prob.vars.rhs
  prob.t += prob.dt
  prob.step += 1
  nothing
end

function stepforward!(prob, nsteps)
  for i = 1:nsteps
    stepforward!(prob)
  end
  nothing
end

function randomIC!(prob; amplitude=1)
  q0 = amplitude*rand(prob.grid.nx, prob.grid.nx)
  q0h = rfft(q0)
  @. prob.vars.qh = q0h
  nothing
end

end # module
