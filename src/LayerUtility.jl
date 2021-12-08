module LayerUtility

export Chain, Activation, BatchNorm, BNorm, Conv, DeConv, deconv_weight_init, conv_weight_init, linlayer_weight_init, mybatchnorm

ENV["COLUMNS"]=72
using Distributions
using Interpolations
using Knet: Knet, dir, accuracy, progress, sgd, load143, save, gc, Param, KnetArray, Data, minibatch, nll, relu, training, dropout,sigm # param, param0, xavier_uniform
using Knet
using Images
using Plots
using LinearAlgebra
using IterTools: ncycle, takenth
using MLDatasets
using Base.Iterators: flatten
using CUDA # functional
using ImageTransformations
using Statistics
# using Interpolations
atype = (CUDA.functional() ? KnetArray{Float32} : Array{Float32})
array_type = atype 

function deconv_weight_init(w1::Int, w2::Int, cx::Int, cy::Int; bias::Bool = true, return_param = false)
    k = sqrt(1/(cy * w1 * w2))
    if return_param
        if bias
            return param(rand(Uniform(-k,k), w1, w2, cy, cx)), param(rand(Uniform(-k,k), 1, 1, cy, 1))
        else
            return  param(rand(Uniform(-k,k), w1, w2, cy, cx))
        end
    else
        if bias
            return (rand(Uniform(-k,k), w1, w2, cy, cx)), (rand(Uniform(-k,k), 1, 1, cy, 1))
        else
            return  (rand(Uniform(-k,k), w1, w2, cy, cx))
        end
    end
end

function conv_weight_init(w1::Int, w2::Int, cx::Int, cy::Int; bias::Bool = true, return_param = false)
    k = sqrt(1/(cx * w1 * w2))
    if return_param
        if bias
            return param(rand(Uniform(-k, k), w1, w2, cx, cy)), param(rand(Uniform(-k, k), 1, 1, cy, 1))
        else
            return param(rand(Uniform(-k, k), w1, w2, cx, cy))
        end
    else
        if bias
            return (rand(Uniform(-k, k), w1, w2, cx, cy)), (rand(Uniform(-k, k), 1, 1, cy, 1))
        else
            return (rand(Uniform(-k, k), w1, w2, cx, cy))
        end
    end
end

function linlayer_weight_init(i::Int, o::Int; bias::Bool = true, return_param = false)
    k = sqrt(1/i)
    if return_param
        if bias
            return param(rand(Uniform(-k, k), o, i)), param(rand(Uniform(-k, k), o, 1))
        else
            return param(rand(Uniform(-k, k), o, i))
    end
    else
        if bias
            return (rand(Uniform(-k, k), o, i)), (rand(Uniform(-k, k), o, 1))
        else
            return (rand(Uniform(-k, k), o, i))
        end
    end
end

# Let's define a chain of layers
struct Chain
    layers
    Chain(layers...) = new(layers)
end

(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

# Activation Layer
mutable struct Activation; f; end;

(a::Activation)(x) = a.f.(x)

# 2D Batch Normalization
mutable struct BatchNorm
    moments
    params
    nParameters
end

(b::BatchNorm)(x) = batchnorm(x, b.moments, b.params)

BatchNorm(C) = BatchNorm(bnmoments(), atype(bnparams(C)), 0)

moments = bnmoments(momentum = 0.1)

# Batch Normalization Layer Definition
mutable struct BNorm
    moments
    params
    act
end

(bn::BNorm)(x) = if bn.act Knet.elu.(batchnorm(x, bn.moments, bn.params)) else batchnorm(x, bn.moments, bn.params) end

# Define a convolutional layer:
struct Conv; w; b; f; p; stride; padding; end

Conv(w1::Int,w2::Int,cx::Int,cy::Int,f=identity; pdrop=0, stride = 1, padding = 0) = 
Conv(param(rand(Uniform(-sqrt(1/(cx * w1 * w2)),sqrt(1/(cx * w1 * w2))), w1,w2,cx,cy)), 
param(rand(Uniform(-sqrt(1/(cx * w1 * w2)),sqrt(1/(cx * w1 * w2))), 1,1,cy,1)), f, pdrop, stride, padding)

(c::Conv)(x) = (conv4(c.w, dropout(x,c.p), padding = c.padding, stride = c.stride, mode = 1) .+ c.b)

# Transposed Convolution Definition
struct DeConv; w; b; f; p; stride; padding; end

DeConv(w1::Int, w2::Int, cx::Int, cy::Int, f = identity; pdrop = 0, stride = 1, padding = 0) = 
DeConv(param(rand(Uniform(-sqrt(1/(cy * w1 * w2)),sqrt(1/(cy * w1 * w2))), w1,w2,cy,cx)), 
    param(rand(Uniform(-sqrt(1/(cy * w1 * w2)),sqrt(1/(cy * w1 * w2))), 1,1,cy,1)), f, pdrop, stride, padding)

(dc::DeConv)(x) = (deconv4(dc.w, dropout(x, dc.p), padding = dc.padding, stride = dc.stride, mode = 1) .+ dc.b)

function mybatchnorm(x, moments, bparam; training = true)
    bparam_dim =  size(bparam,1)
    g = reshape(bparam[1:bparam_dim/2], (1,1,Int(bparam_dim/2), 1))
    b = reshape(bparam[bparam_dim/2 + 1 : bparam_dim], (1,1,Int(bparam_dim/2), 1))
    return g.* batchnorm(x, moments; training = training) .+ b
end

end