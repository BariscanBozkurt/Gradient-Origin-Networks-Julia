# Set display width, load packages, import symbols
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
import CUDA # functional
using ImageTransformations
using Statistics
using Memento
using NPZ
# using Interpolations
atype=(CUDA.functional() ? KnetArray{Float32} : Array{Float32})

const F = Float32
params = Knet.params

logger = Memento.config!("info"; fmt="[{date} | {level} | {name}]: {msg}");

include("PlotUtility.jl")
include("ImageUtility.jl")
include("TrainUtility.jl")
include("LayerUtility.jl")
include("LossUtility.jl")

using .PlotUtility
using .ImageUtility
using .TrainUtility
using .LayerUtility
using .LossUtility

########################### CHANGE THIS LINE FOR DATASET PARAMETER ##############################
dataset_name = "cifar"
exp_number = 1
########################### CHANGE THIS LINE FOR RESULT FOLDER NAME #############################
notebook_name = "GON_Autoencoder_ngf32_nz256" * "_" * dataset_name * string(exp_number)

if !isdir("Results")
   mkdir("Results") 
end
if  !isdir(joinpath("Results", notebook_name))
    mkdir(joinpath("Results", notebook_name))
end

if  !isdir(joinpath("Results", notebook_name, "Saved_Models"))
    mkdir(joinpath("Results", notebook_name, "Saved_Models"))
end

if  !isdir(joinpath("Results", notebook_name, "Images"))
    mkdir(joinpath("Results", notebook_name, "Images"))
end

if  !isdir(joinpath("Results", notebook_name, "Logger"))
    mkdir(joinpath("Results", notebook_name, "Logger"))
end

push!(logger, DefaultHandler(joinpath("Results", notebook_name, "Logger", "logger.log"),DefaultFormatter("[{date} | {level} | {name}]: {msg}")));

use_saved_data = true
nc = nothing

if dataset_name == "mnist"
    nc = 1
    if use_saved_data
        xtrn = npzread("Data/MNIST_Train_Images.npy")
        xtrn = permutedims(xtrn, (3,4,2,1))
        xtst = npzread("Data/MNIST_Test_Images.npy")
        xtst = permutedims(xtst, (3,4,2,1))

    else

        xtrn,_ = MNIST.traindata()
        xtst,_ = MNIST.testdata()
        xtrn = Array{Float64, 3}(xtrn)
        xtst = Array{Float64, 3}(xtst)

        xtrn = resize_MNIST(xtrn, 32/28)
        xtst = resize_MNIST(xtst, 32/28)
        
    end
    
elseif dataset_name == "fashion"
    nc = 1
    if use_saved_data

        xtrn = npzread("Data/Fashion_MNIST_Train_Images.npy")
        xtrn = permutedims(xtrn, (3,4,2,1))
        xtst = npzread("Data/Fashion_MNIST_Test_Images.npy")
        xtst = permutedims(xtst, (3,4,2,1))
        
    else
        
        xtrn,_ = FashionMNIST.traindata()
        xtst,_ = FashionMNIST.testdata()
        xtrn = Array{Float64, 3}(xtrn)
        xtst = Array{Float64, 3}(xtst)

        xtrn = resize_MNIST(xtrn, 32/28)
        xtst = resize_MNIST(xtst, 32/28)

    end
    
elseif dataset_name == "cifar"
    nc = 3
    xtrn,_= CIFAR10.traindata()
    xtst,_ = CIFAR10.testdata()
    xtrn = Array{Float64, 4}(xtrn)
    xtst = Array{Float64, 4}(xtst)
#     println("No implemented yet")
end

batch_size = 64

dtrn = minibatch(xtrn, batch_size; xsize = (32, 32, nc,:), xtype = atype, shuffle = true)
dtst = minibatch(xtst, batch_size; xsize = (32, 32, nc,:), xtype = atype);

# Plotting some images if using jupyter notebook
k = 1:10
using_jupyter = false
if using_jupyter
    if (dataset_name == "fashion") || (dataset_name == "mnist")
        if use_saved_data
            [Matrix{Gray{Float32}}(reshape(xtrn[:,:,:,j], (32, 32))) for j in k]
        else
            [Matrix{Gray{Float32}}(reshape(xtrn[:,:,j], (32, 32))) for j in k]
        end
    end
end

function weights(nc, nz,  ngf) 
        
    # Decoding Weights
    theta = [] # z -> x
    
    w, b = deconv_weight_init(4, 4, nz, ngf * 4;bias = true, return_param = false)
    push!(theta, w)
    push!(theta, b)
    push!(theta, (bnparams(ngf * 4))) # Batch Normalization
    
    w, b = deconv_weight_init(4, 4, ngf * 4, ngf * 2;bias= true, return_param = false)
    push!(theta, w)
    push!(theta, b)
    push!(theta, (bnparams(ngf * 2)))
    
    w, b = deconv_weight_init(4, 4, ngf * 2, ngf;bias = true, return_param = false)
    push!(theta, w)
    push!(theta, b)
    push!(theta, (bnparams(ngf)))
    
    w,b = deconv_weight_init(4,4,ngf, nc;bias = true, return_param = false)
    push!(theta, w)
    push!(theta, b)

    theta = map(a->convert(atype,a), theta)
    
    return Param.(theta)
end

moments1 = bnmoments()
moments2 = bnmoments()
moments3 = bnmoments()

function decode(theta, z; batch_size = 64, training = true)
    
    z = deconv4(theta[1], z, mode = 1) .+ theta[2]
    z = mybatchnorm(z, moments1, theta[3]; training = training)
    z = Knet.elu.(z)
    
    z = deconv4(theta[4], z, mode = 1, stride = 2, padding = 1) .+ theta[5]
    z = mybatchnorm(z, moments2, theta[6]; training = training)
    z = Knet.elu.(z)
    
    z = deconv4(theta[7], z, mode = 1, stride = 2, padding = 1) .+ theta[8]
    z = mybatchnorm(z, moments3, theta[9]; training = training)
    z = Knet.elu.(z)
    
    z = deconv4(theta[10], z, mode = 1, stride = 2, padding = 1) .+ theta[11]
    x_hat = Knet.sigm.(z)
    return x_hat
    
end

function decode_train(theta, x; batch_size = 64)
    origin = Param(atype(zeros(1, 1, nz, batch_size)))
    derivative_origin = @diff inner_loss(x,origin)
    dz = grad(derivative_origin, origin)
    z = -dz
    x_hat = decode(theta, z; batch_size = batch_size, training = true)
    return x_hat
end

# (Reconstruction Error := Mean Squared Error)
inner_loss(x, z) = mean(sum((decode(theta, z; batch_size = size(x,4)) - x).^2, dims = 3))

function loss_train(theta, x)
    x_hat =  decode_train(theta, x; batch_size = size(x,4))
    L = mean(sum((x_hat - x).^2, dims = 3))
end

function loss_train(theta, d::Data)
    total_loss = 0
    n_instance = 0
    for x in d
        total_loss += loss_train(theta, x) * size(x,4)
        n_instance += size(x,4)
    end
    
    total_loss /= n_instance
end


# Sum Squared Error Definition
rec_loss(theta, x) = sum((x - decode_train(theta, x; batch_size = size(x,4))).^2) / size(x,4)

function rec_loss(theta, d::Data)
    total_loss = 0
    n_instance = 0
    for x in d
        total_loss += rec_loss(theta, x) * size(x,4)
        n_instance += size(x,4)
    end
    
    total_loss /= n_instance
end

############################## CHANGE THE FOLLOWING LINES FOR LATENT DIM AND NUMBER OF FILTERS ############################
nz = 256
ngf = 32
# nc = 1

# first batch of the test dataset
x_test_first = first(dtst);

# define model weights
theta = weights(nc, nz,  ngf);

# Define Learning Rate and Number of Epochs
lr = 2*1e-4
n_epochs = 500

# optim = Knet.Adam(lr = lr, beta1 = 0.9, beta2 = 0.999)

# Specify the optimizer for each param
for p in params(theta)
    p.opt =  Knet.Adam(lr = lr, beta1 = 0.9, beta2 = 0.999)
end
# Initialize Empty Lists for both training and test losses
trn_loss_list = Float64[]
tst_loss_list = Float64[]
trn_rec_loss_list = Float64[]
tst_rec_loss_list = Float64[]

# RECORD INITIAL LOSS VALUES
epoch_loss_trn_ = loss_train(theta, dtrn)
epoch_loss_tst_ = loss_train(theta, dtst)
epoch_rec_loss_trn_ = rec_loss(theta, dtrn)
epoch_rec_loss_tst_ = rec_loss(theta, dtst)

push!(trn_loss_list, epoch_loss_trn_)
push!(tst_loss_list, epoch_loss_tst_)
push!(trn_rec_loss_list, epoch_rec_loss_trn_)
push!(tst_rec_loss_list, epoch_rec_loss_tst_)

# println("Epoch : ", 0)
# println("Train Loss : ",epoch_loss_trn_)
# println("Test Loss : ", epoch_loss_tst_)
# println("Train Reconstruction Loss : ", epoch_rec_loss_trn_)
# println("Test Reconstruction Loss : ", epoch_rec_loss_tst_)

info(logger, ("Now training is starting. We provide the parameters as the following"))
info(logger, "Dataset = $dataset_name")
info(logger,"nz = $nz")
info(logger,"ngf = $ngf")
info(logger, "nc = $nc")
info(logger, "lr = $lr")
info(logger, "n_epochs = $n_epochs")


info(logger, ("Epoch : 0"))
info(logger, ("Train Loss : $epoch_loss_trn_"))
info(logger, ("Test Loss : $epoch_loss_tst_"))
info(logger, ("Train Reconstruction Loss : $epoch_rec_loss_trn_"))
info(logger, ("Test Reconstruction Loss : $epoch_rec_loss_tst_ \n"))

########################################## CHANGE THE FOLLOWING LINES FOR CHECKPOINT ITERATION NUMBERS ############################
# Define the step number of model save checkpoint
model_save_checkpoint = 50
logger_checkpoint = 1
image_rec_checkpoint = 50

# Training Loop
for epoch in progress(1:n_epochs)
    for (i,x) in enumerate(dtrn)
        
        derivative_model = @diff loss_train(theta, x)
        
        for p in theta
            dp = grad(derivative_model, p)
            update!(value(p), dp, p.opt)
        end

    end
    
    epoch_loss_trn = loss_train(theta, dtrn)
    epoch_loss_tst = loss_train(theta, dtst)
    epoch_rec_loss_trn = rec_loss(theta, dtrn)
    epoch_rec_loss_tst = rec_loss(theta, dtst)
    
    push!(trn_loss_list, epoch_loss_trn)
    push!(tst_loss_list, epoch_loss_tst)
    push!(trn_rec_loss_list, epoch_rec_loss_trn)
    push!(tst_rec_loss_list, epoch_rec_loss_tst)
    
#     println("Epoch : ", epoch)
#     println("Train Loss : ",epoch_loss_trn)
#     println("Test Loss : ", epoch_loss_tst)
#     println("Train Reconstruction Loss : ", epoch_rec_loss_trn)
#     println("Test Reconstruction Loss : ", epoch_rec_loss_tst)
    
    # Print losses to the logger file
    if epoch % logger_checkpoint == 0
        info(logger,"Epoch : $epoch")
        info(logger,"Train Loss : $epoch_loss_trn")
        info(logger,"Test Loss : $epoch_loss_tst")
        info(logger,"Train Reconstruction Loss : $epoch_rec_loss_trn")
        info(logger,"Test Reconstruction Loss : $epoch_rec_loss_tst \n")
    end
    
    # # Plot some reconstruction images
    # if ((epoch - 1) % image_rec_checkpoint == 0) || (epoch == n_epochs)
    #     if (dataset_name == "mnist") ||(dataset_name == "fashion")
    #         x_test_first_hat = decode_train(theta, x_test_first)
    #         plot_reconstructed_images(x_test_first, x_test_first_hat, 10, batch_size, (900,300))
    #         fig_name = "Reconstructed_Imgs_ID" * string(1000 + epoch) 
    #         savefig(joinpath("Results", notebook_name, "Images", fig_name))
    #     end
    # end
    
    # Save model at some steps
    if (epoch % model_save_checkpoint == 0) || (epoch == n_epochs)
        
        model_id = 1000 + epoch
        model_name = joinpath("Results", notebook_name, "Saved_Models","Model_Base$model_id.jld2")
        w = Dict(:decoder => theta)
        Knet.save(model_name,"model",w) 
        ### TO LOAD THE MODEL WEIGHTS, USE THE FOLLOWING
        # w = Knet.load(model_name,"model",) # Ex: model_name = "Results/Conv_AutoEncoder_Baseline_MNIST/Saved_Models/Model_Base1500.jld2"
        # theta = w[:decoder]
        
    end
    
end

info(logger, ("Now training is done. Recall the parameters as the following"))
info(logger, "Dataset = $dataset_name")
info(logger,"nz = $nz")
info(logger,"ngf = $ngf")
info(logger, "nc = $nc")
info(logger, "lr = $lr")
info(logger, "n_epochs = $n_epochs")

# plot_loss_convergence(trn_loss_list[2:end], tst_loss_list[2:end]; title = "Train & Test Loss w.r.t. Epochs")
# fig_name = "Train_and_test_loss"
# savefig(joinpath("Results", notebook_name, fig_name))

# plot_loss_convergence(trn_rec_loss_list[2:end], tst_rec_loss_list[2:end]; title = "Train & Test Reconstruction Loss w.r.t. Epochs")
# fig_name = "Train_and_test_reconstruction_loss"
# savefig(joinpath("Results", notebook_name, fig_name))

info(logger, "Training is done!")
info(logger, "We will report the last loss values for both training and test sets.\n")

epoch_loss_trn = loss_train(theta, dtrn)
epoch_loss_tst = loss_train(theta, dtst)
epoch_rec_loss_trn = rec_loss(theta, dtrn)
epoch_rec_loss_tst = rec_loss(theta, dtst)

info(logger,"Train Loss : $epoch_loss_trn")
info(logger,"Test Loss : $epoch_loss_tst")
info(logger,"Train Reconstruction Loss : $epoch_rec_loss_trn")
info(logger,"Test Reconstruction Loss : $epoch_rec_loss_tst \n")

Knet.save(joinpath("Results", notebook_name,"trn_loss_list.jld2"),"trn_loss_list",trn_loss_list) 
Knet.save(joinpath("Results", notebook_name,"tst_loss_list.jld2"),"tst_loss_list",tst_loss_list) 
Knet.save(joinpath("Results", notebook_name,"trn_rec_loss_list.jld2"),"trn_rec_loss_list",trn_rec_loss_list) 
Knet.save(joinpath("Results", notebook_name,"tst_rec_loss_list.jld2"),"tst_rec_loss_list",tst_rec_loss_list) 


