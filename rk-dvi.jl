# code for Runge-Kutta solution of differential complementarity problems of index one.


module RKDCP

# general functions
export RKmethod, DCP, nstages, rk, rk_dcp_eqns, rk_dcp, drk_dcp_eqns # rk_explicit, rk_implicit # , rkfunc, drkfunc
export expEuler, impEuler, heun, trap, rk4, radauIIA2, radauIIA3, dirkA, dirkCR

# require("newton.jl")

# using Newton

"""
Runge-Kutta method represented by Butcher tableau.
T is the floating point type
"""
struct RKmethod{T}
  name::String
  A :: Array{T,2}
  b :: Array{T,1}
  c :: Array{T,1}
  explicit :: Bool
end

# various RK methods
expEuler = RKmethod("explicit Euler",
    zeros(1,1),[1.0],[0.0], true)
impEuler = RKmethod("implicit Euler", 
    ones(1,1),[1.0],[1.0], false)
heun     = RKmethod("Heun's method",
    [0 0; 1 0.0],[1/2;1/2],[0;1.0], true)
trap     = RKmethod("implicit trapezoidal method",
    [0 0; 1/2 1/2],[1/2;1/2],[0;1.0], false)
rk4      = RKmethod("standard 4th order RK",[0 0 0 0; 1/2 0 0 0; 0 1/2 0 0; 0 0 1.0 0],
                           [1; 2; 2; 1]/6.0, [0.0;1/2;1/2;1.0], true)
radauIIA2 = RKmethod("RadauIIA (2-stage)",[5/12 -1/12; 3/4 1/4],[3/4;1/4],[1/3,1], false)
radauIIA3 = RKmethod("RadauIIA (3-stage)",[(88-7sqrt(6))/360 (296-169sqrt(6))/1800 (-2+3sqrt(6))/225;
                      (296+169sqrt(6))/1800 (88+7sqrt(6))/360 (-2-3sqrt(6))/225;
                      (16-sqrt(6))/36 (16+sqrt(6))/36 1/9],
                     [(16-sqrt(6))/36; (16+sqrt(6))/36; 1/9],
                     [(4-sqrt(6))/10; (4+sqrt(6))/10; 1], false)
# DIRK methods of Alexander (dirkA), and Crouzeix & Raviart (dirkCR)
theta = (1/3)*atan(2^(-3/2))
alpha = cos(theta)/sqrt(2) + 1
tau2 = (1/2)*(1+alpha)
b1 = -(1/4)*(6alpha^2-16alpha+1)
b2 =  (1/4)*(6alpha^2-20alpha+5)
dirkA = RKmethod("Alexander DIRK",
    [alpha 0 0; tau2-alpha alpha 0; b1 b2 alpha],
		 [b1;b2;alpha],[alpha;tau2;1], false)
gamma = (1/sqrt(3))*cos(pi/18)+1/2
delta = 1/(6*(2gamma-1)^2)
dirkCR = RKmethod("Crouzeix & Raviart DIRK",
    [gamma 0 0; 1/2-gamma gamma 0; 2gamma 1-4gamma gamma],
		  [delta; 1-2delta; delta],[gamma; 1/2; 1-gamma], false)

nstages(rk::RKmethod{T}) where {T} = size(rk.A,1)

using LinearAlgebra


"""
    struct DCP
A convenient way of keeping all the functions `f`, `dfdv`, `dfdz`, `G`, and `dG` together.
"""
struct DCP
    f::Function
    dfdv::Function
    dfdz::Function
    G::Function
    dG::Function
end


"""
    rk_dcp_eqns(f::Function, G::Function, rk::RKmethod{T}, h::T, t::T, x::Vector{T}, vz_vec::Vector{T}) where {T}
Returns 
``v_i - x - h\\sum_j a_{ij} f(t,v_j,z_j)`` for ``j=1,2,\\ldots,s`` followed by
``\\min(z_i, G(v_i))`` for ``j = 1,\\ldots,s``.

Each ``v_i`` is a vector of length ``n`` (where dimension of ``x`` is ``n``), and each ``z_i`` is a vector of length ``m`` (where dimension of ``G(x)`` is ``m``). Also ``s`` is the number of stages in method `rk`.

The `vz_vec` has length ``s(n+m)`` where ``v_i`` is in `vz_vec[(i-1)*n+1:i*n]` and ``z_i`` is in `vz_vec[n*s+(i-1)*m+1:n*s+i*m]`.
The output vector has the same length as `vz_vec` and the corresponding layout.
"""
function rk_dcp_eqns(dcp::DCP, rk::RKmethod{T}, h::T, t::T, x::Vector{T}, vz_vec::Vector{T}) where {T}
    f = dcp.f; G = dcp.G
    s = nstages(rk)
    out = zeros(T,length(vz_vec))
    n = length(x)
    m = length(G(x))
    v = Array{Vector{T},1}(undef,s)
    z = Array{Vector{T},1}(undef,s)
    for i = 1:s
        v[i] = vz_vec[((i-1)*n+1):(i*n)]
        z[i] = vz_vec[(n*s+(i-1)*m+1):(n*s+i*m)]
    end
    for i = 1:s
        out[((i-1)*n+1):(i*n)] = v[i] - x
        for j = 1:s
            out[((i-1)*n+1):(i*n)] -= h*rk.A[i,j]*f(t+rk.c[j]*h,v[j],z[j])
        end
        out[(n*s+(i-1)*m+1):(n*s+i*m)] = min.(z[i],G(v[i]))
    end
    return out
end

"""
    drk_dcp_eqns(dcp::DCP, rk::RKmethod{T}, h::T, t::T, vz_vec::Vector{T}) where {T}
Jacobian matrix of `rk_dcp_eqns` w.r.t. `vz_vec`
"""
function drk_dcp_eqns(dcp::DCP, rk::RKmethod{T}, h::T, t::T, x::Vector{T}, vz_vec::Vector{T}) where {T}
    f = dcp.f; G = dcp.G; dG = dcp.dG
    dfdv = dcp.dfdv; dfdz = dcp.dfdz
    J = zeros(T,length(vz_vec),length(vz_vec))
    n = length(x)
    m = length(G(x))
    s = nstages(rk)
    v = Array{Vector{T},1}(undef,s)
    z = Array{Vector{T},1}(undef,s)
    vidxs = Array{Vector{Int},1}(undef,s)
    zidxs = Array{Vector{Int},1}(undef,s)
    for i = 1:s
        vidxs[i] = [((i-1)*n+1):(i*n);]
        zidxs[i] = [(n*s+(i-1)*m+1):(n*s+i*m);]
        v[i] = vz_vec[vidxs[i]]
        z[i] = vz_vec[zidxs[i]]
    end
    for i = 1:s
        J[vidxs[i],vidxs[i]] = I(n)
        for j = 1:s
            J[vidxs[i],vidxs[j]] .-= h*rk.A[i,j]*dfdv(t+rk.c[j]*h,v[j],z[j])
            J[vidxs[i],zidxs[j]] .-= h*rk.A[i,j]*dfdz(t+rk.c[j]*h,v[j],z[j])
        end
        w = G(v[i])
        J[zidxs[i],vidxs[i]] = Diagonal(T.(z[i] .> w))*dG(v[i])
        J[zidxs[i],zidxs[i]] = Diagonal(T.(z[i] .< w))
    end
    J
end



"""
    rk_dcp(dcp::DCP, rk::RKmethod{T}, t0::T, x0::Vector{T}, z0::Vector{T}, h::T, nsteps::Int; epsilon::T=T(1e-6)) where {T}
Returns discrete trajectory for the DCP defined by `dcp` using the Runge-Kutta method `rk`.
The initial state is `x0` and initial time `t0`. The error tolerance `epsilon` is used for solving the DCP RK equations.
A fixed step size `h` is used. An initial guess `z0` for `z` is passed.
"""
function rk_dcp(dcp::DCP, rk::RKmethod{T}, t0::T, x0::Vector{T}, z0::Vector{T}, h::T, nsteps::Int; 
            epsilon::T=T(1e-6)) where {T}
    n = length(x0)
    m = length(z0)
    s = nstages(rk)
    vz = zeros(T,(n+m)*s)
    x = copy(x0)
    z = copy(z0)
    xlist = [x]
    tlist = [t0]
    zlist = Array{Vector{T}}(undef,0)
    vidxs = Array{Vector{Int}}(undef,s)
    zidxs = Array{Vector{Int}}(undef,s)
    for i = 1:s
        vidxs[i] = [((i-1)*n+1):(i*n);]
        zidxs[i] = [(n*s+(i-1)*m+1):(n*s+i*m);]
    end
    t = t0

    for k = 1:nsteps
        # println("step ",k, ", vz = ", vz)
        # set up initial guess
        for i = 1:s
            vz[vidxs[i]] .= x
            vz[zidxs[i]] .= z
        end
        
        # solve RK eqn's
        fvz = rk_dcp_eqns(dcp, rk, h, t, x, vz)
        # println("rhs = ",fvz)
        done = false
        while ! done
            # println("Newton's method")
            if norm(fvz) < epsilon*h
                done = true
                break
            end
            dfvz = drk_dcp_eqns(dcp, rk, h, t, x, vz)
            # println("dfvz = ",dfvz)
            dvz = - dfvz \ fvz # Newton step
            # we'll insert the line search if needed
            vz += dvz # update vz
            # println("updated vz = ",vz)
            fvz = rk_dcp_eqns(dcp, rk, h, t, x, vz) # & update fvz
            # println("updated fvz = ",fvz)
            # println()
        end
        
        # update and save x, z, etc.
        # use stiffly accurate property
        x = vz[vidxs[s]]
        z = vz[zidxs[s]]
        # println("x = ",x)
        # println("z = ",z)
        push!(xlist,copy(x))
        push!(zlist,copy(z))
        t += h
        push!(tlist,t)
    end
    
    return xlist, zlist, tlist
end


function rk_default_solver(f::Function,df::Function,rk::RKmethod{T},t::T,x::V,h::T,vlist::Array{V,1},eps::T) where {T,V}
	s = size(rk.A,1)
	max_err = zero(T)
	for i = 2:s
		xtemp = copy(x)
		for j = 1:i-1
			xtemp += (h*rk.A[i,j])*vlist[j]
		end
		ftemp = f(t+rk.c[i]*h, xtemp)
		norm_err = norm(ftemp-vlist[i])
		max_err = max(max_err,norm_err)
		vlist[i] = ftemp
	end
	# fixed-point iteration 
	while (! rk.explicit && max_err > eps)
		max_err = zero(T)
		for i = 1:s
			xtemp = copy(x)
			for j = 1:i-1
				xtemp += (h*rk.A[i,j])*vlist[j]
			end
			ftemp = f(t+rk.c[i]*h, xtemp)
			norm_err = norm(ftemp-vlist[i])
			max_err = max(max_err,norm_err)
			vlist[i] = ftemp
		end
	end
	return vlist
end


#=
"""
	rk -- solves ODE using Runge-Kutta method
	solves
		dx/dt = f(t,x),  x(t0) = x0 in V
	T is an approximate real field
	V is approximate vector spaces over T
	solver solves the RK equations
		v_i = f(t+c_i h, x + h\sum_{j=1}^s a_{ij}v_j), i = 1,2,..,s
	eps is the tolerance for solver().
"""
function rk_method(f::Function,x0::V,rk::RKmethod{T},
	t0::T,h::T,n::Int,eps::T;
	# solve_init::Function = rk_default_init,
	solver::Function = rk_default_solver,df) where {T,V,W}
	# start of function
	s = size(rk.A,1)
	t = t0; x = x0;
	tlist = [t0]; xlist = [x0]
	for step = 1:n # for each step
		# current point (x,t)
		# initialize solver
		vinit = f(t,x)
		vlist = fill(vinit,(s,)) # creates vlist[i] = vinit for all i
		vlist = solver(f,df,rk,t,x,h,vlist,eps)
		for i = 1:s
			x = x + (h*rk.b[i])*vlist[i]
		end
		t += h
		append(tlist,t)
		append(xlist,x)
	end
	return (xlist,tlist)
end


"""
	rk_dae -- solves DAE using Runge-Kutta method
	solves
		dx/dt = f(t,x,y),  x(t0) = x0 in V
		0     = g(t,x,y),  y(t0) = y0 in W
	T is an approximate real field
	V and W are approximate vector spaces over T
	solver
"""
function rk_dae(f::Function,g::Function,x0::V,y0::W,rk::RKmethod{T},
	t0::T,h::T,n::Int;solver::Function = rk_default_solver) where {T,V,W}
end
=#

end # module RKDCP
