module TrainUtility

export train_model_w_ADAM


# Set display width, load packages, import symbols
ENV["COLUMNS"]=72
# using Knet: Knet, dir, accuracy, progress, sgd, load143, save, gc, Param, KnetArray, Data, minibatch, nll, relu, training, dropout,sigm # param, param0, xavier_uniform
using Knet
using LinearAlgebra
using IterTools: ncycle, takenth
using Base.Iterators: flatten
using CUDA

atype = (CUDA.functional() ? KnetArray{Float32} : Array{Float32})
params = Knet.params

function train_model_w_ADAM(m, L, dtrn, n_epochs, lr; verbose = true, watch_tst_loss = false, dtst = nothing)
    batch_loss_list = Float64[]
    loss_list = Float64[]
    tst_loss_list = Float64[]
    
    if verbose
        for epoch in (1:n_epochs)
            for x in progress(dtrn )
                x_rec = m(x)
                push!(batch_loss_list, L(m,x))
                derivative = @diff L(m,x)
                for p in params(m)
                     dp = grad(derivative, p)
                     update!(value(p), dp, Knet.Adam(lr = lr,beta1 = 0.9, beta2 = 0.999))
                end
            end
            push!(loss_list, L(m, dtrn))

            last_loss = loss_list[end]

            
            if watch_tst_loss
                if dtst == nothing
                    print("Epoch: $epoch, Train Loss: $last_loss")
                else
                    last_tst_loss = L(m, dtst)
                    push!(tst_loss_list, last_tst_loss)
                    print("Epoch: $epoch, Train Loss: $last_loss, Test Loss: $last_tst_loss")
                end
            end
            
        end
        
    else
        for epoch in progress(1:n_epochs)
            for x in dtrn 
                x_rec = m(x)
                push!(batch_loss_list, L(m,x))
                derivative = @diff L(m,x)
                for p in params(m)
                        dp = grad(derivative, p)
                        update!(value(p), dp, Knet.Adam(lr = lr,beta1 = 0.9, beta2 = 0.999)pt)
                end
            end
            push!(loss_list, L(m, dtrn))

            last_loss = loss_list[end]
            if watch_tst_loss
                if dtst == nothing
                    continue
                else
                    push!(tst_loss_list, L(m, dtst))
                end
            end
        end

    end
    return batch_loss_list, loss_list, tst_loss_list
end

end