using 
  SimpleNavierStokes,
  Printf,
  PyPlot,
  FFTW

setup = Setup(T=Float64, nx=512, Lx=2π, nu=1e-5, dt=1e-1)
prob = Problem(setup)
randomIC!(prob)

nsteps = 100
njumps = 20

# Step forward
close("all")
fig = figure()

for step = 1:njumps

  tcomp = @time begin
    stepforward!(prob, nsteps)
    @printf "step: %04d, t: %6.1f, " prob.step prob.t
  end

  qh = deepcopy(prob.vars.qh)
  q = irfft(qh, prob.grid.nx)

  cla()
  imshow(q)
  pause(0.01)
end
