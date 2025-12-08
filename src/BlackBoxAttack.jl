struct BlackBoxAttack <: AbstractAttack
    parameters::Dict{String,Any}

    function BlackBoxAttack(parameters::Dict{String,Any}=Dict{String,Any}()) # why =Dict{String,Any}()?
        new(parameters)
    end
end

function perform_attack(attack::BlackBoxAttack, model, sample)
    return sample
end