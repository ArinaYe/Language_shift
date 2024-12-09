@time begin
include("GridVL.jl")
using InteractiveDynamics, Agents
using .GridVL
using StatsBase
using Distributions 
using CairoMakie
using BenchmarkTools

dims = (105, 64)
space = GridSpaceSingle(dims, periodic=true)

model = StandardABM(VariationalLearner, space; agent_step! = VL_step!)

for i in 1:6720
    probabilities = [0.023, 0.067, 0.909]
    normalized_probs = probabilities / sum(probabilities)
    states = [0, 1, 2]
    dist = Categorical(normalized_probs)  
    state_idx = rand(dist)
    state = states[state_idx]  
    add_agent_single!(model; state=state)
end

# initialize_agents!(model, dims)
function getstate(a)
    return a.state
end

fig, meta = abmplot(model; ac=getstate, as=8)
fig

step!(model, 5)
fig1, meta = abmplot(model; ac=getstate, as=8)
fig1

abmvideo("video2.mp4", model; agent_color = getstate, as = 8, 
         frames = 10, framerate = 1)

end