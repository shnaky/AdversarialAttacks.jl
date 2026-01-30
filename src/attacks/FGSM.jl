"""
    FGSM(; epsilon=0.1)

Subtype of `WhiteBoxAttack`. A struct that can be used to create a White-box Adversarial Attack using the Fast Gradient Sign Method.
Based on Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). [Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572). arXiv preprint arXiv:1412.6572.



# Arguments
- `epsilon`: Step size used to scale the sign of the gradient. Defaults to `0.1`.
"""
struct FGSM{T <: Real} <: WhiteBoxAttack
    epsilon::T
end

FGSM(; epsilon::Real = 0.1) = FGSM(epsilon)

default_loss(m, x, y) = crossentropy(m(x), y)

"""
    attack(atk::FGSM, model, sample; loss, detailed_result)

Perform a Fast Gradient Sign Method (FGSM) White-box Adversarial Attack on the given `model` using the provided `sample`.

# Arguments
- `atk::FGSM`: An instance of the `FGSM`.
- `model::FluxModel`: The machine learning (deep learning) model to be attacked.
- `sample`: Input sample as a named tuple with `data` and `label`.
- `loss`: Loss function with signature `loss(model, x, y)`. Defaults to `default_loss`, i.e. cross-entropy.
- `detailed_result::Bool=false`: Return format control.
  - `false` (default): Returns adversarial example only (Array).
  - `true`: Returns NamedTuple with metrics (x_adv, success, queries_used, final_label).

# Returns
- If `detailed_result=false`:
    - Adversarial example (same type and shape as `sample.data`).
- If `detailed_result=true`:
    - NamedTuple with fields:
        - `x_adv`: Adversarial example.
        - `queries_used::Int`: Number of gradient evaluations (for FGSM == 1).
"""
function attack(atk::FGSM, model::Chain, sample; loss = default_loss, detailed_result = false)
    x = sample.data
    y = sample.label
    ε = convert(eltype(x), atk.epsilon)

    # Compute gradient of loss w.r.t. input x
    grads = gradient(xx -> loss(model, xx, y), x)[1]

    # FGSM perturbation: ε · sign(∇_x L(x, y))
    perturbation = ε .* sign.(grads)
    x_adv = x .+ perturbation

    if detailed_result
        return (
            x_adv = x_adv,
            queries_used = 1, # 1 gradient step
        )
    else
        return x_adv
    end
end
