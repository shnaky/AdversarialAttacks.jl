
module Attack


struct AbstractAttack
  name::String
  parameters::Dict{String,Any}

  function AbstractAttack(name::String; parameters::Dict{String,Any}=Dict{String,Any}())
    new(name, parameters)
  end

end


struct WhiteBoxAttack <: AbstractAttack

end


struct BlackBoxAttack <: AbstractAttack

end


end
