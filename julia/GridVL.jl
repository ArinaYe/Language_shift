module GridVL

using Agents
using StatsBase

export VariationalLearner
export VL_step!

@agent struct VariationalLearner(GridAgent{2})
    state::Int64
end

function VL_step!(agent, model)
    update_language_state(agent, model)
end

function social_pressure(agent, model)
    neighbors = nearby_agents(agent, model)
    return sum(neighbor.state for neighbor in neighbors)
end

function update_language_state(agent, model)
    pressure = social_pressure(agent, model)
    S_a, S_b, S_c = 3, 9, 14

    if agent.state == 0
        agent.state = pressure > S_b ? 1 : 0
    elseif agent.state == 1
        if pressure < S_b
            agent.state = 0
        elseif S_b <= pressure && pressure <= S_c
            agent.state = 1
        else
            agent.state = 2
        end
    elseif agent.state == 2
        if pressure <= S_a
            agent.state = 0
        elseif S_a < pressure && pressure <= S_b
            agent.state = 2
        else
            agent.state = 1
        end
    end
end

end