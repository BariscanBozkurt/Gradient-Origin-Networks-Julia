module LossUtility 

export BCE, KLD

using LinearAlgebra
using Knet
using Statistics

const F = Float32
params = Knet.params

function BCE(x_tensor,x_hat_tensor)
    x = mat(x_tensor)
    x_hat = mat(x_hat_tensor)
    return -mean(sum((x .* log.(x_hat .+ F(1e-10)) + (1 .- x) .* log.(1 .- x_hat .+ F(1e-10))), dims = 1))
end

function KLD(mu, logvar)
    # var = exp.(logvar) # This line is not necessary
    # std = sqrt.(var)
    KL = -0.5 * mean(sum(1 .+ logvar .- (mu .* mu) - exp.(logvar), dims = 1))
    return KL
end


end
