struct WhiteBoxAttack <: AbstractAttack
    parameters::Dict{String,Any}

    function WhiteBoxAttack(parameters::Dict{String,Any}=Dict{String,Any}()) # why =Dict{String,Any}()?
        new(parameters)
    end
end

function perform_attack(attack::WhiteBoxAttack, model, sample)
    return sample
end