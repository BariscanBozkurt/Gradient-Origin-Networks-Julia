module PlotUtility

export plot_reconstructed_images, plot_loss_convergence, plot_error_curve, plot_image_grid, plot_weight_histograms

using Images
using Plots
using LinearAlgebra
using IterTools: ncycle, takenth
using ImageTransformations
using Base.Iterators: flatten
using CUDA
using Knet 

atype = (CUDA.functional() ? KnetArray{Float32} : Array{Float32})


function plot_reconstructed_images(im_ori, im_rec, n_instances = 10, max_instance = 64, plot_size = (900,300))
    k = rand(1:max_instance, n_instances)
    ori_plot_list = reshape(im_ori[:,:,:,k[1]], (32, 32))
    recon_plot_list = reshape(im_rec[:,:,:,k[1]], (32, 32))
    for j in k[2:end]
        ori_plot_list = hcat(ori_plot_list, reshape(im_ori[:,:,:,j], (32, 32)))
        recon_plot_list = hcat(recon_plot_list, reshape(im_rec[:,:,:,j], (32, 32)))
    end
    p1 = plot(Matrix{Gray{Float32}}(ori_plot_list), title = "Original Images", size = (20,200),font =  "Courier", xtick = false, ytick = false)
    p2 = plot(Matrix{Gray{Float32}}(recon_plot_list), title = "Reconstructed Images", font = "Courier", xtick = false, ytick = false)
    plot(p1, p2, layout = (2,1), size = (900,300))
end


function plot_loss_convergence(trn_loss, tst_loss = nothing; title = "Train Loss w.r.t. Epochs", figsize = (700, 500), epochs = nothing)
    if epochs == nothing
         epochs = 1:size(trn_loss,1)
    end    
    if tst_loss != nothing
        if size(trn_loss,1) != size(tst_loss,1)
            print("The loss vectors for train and test do not have the same size")
            return 0
        else
            plot(epochs,trn_loss ,label = "Train Loss", xlabel = "Epochs", ylabel = "Loss",
            title = title, minorgrid = true, xtickfontsize = 18,
            ytickfontsize = 18, legendfontsize = 18, xguidefontsize=18, 
            yguidefontsize=18, titlefontsize = 20, size = figsize)
            plot!(epochs,tst_loss ,label = "Test Loss", title = title)
        end
    else
        plot(epochs,trn_loss ,label = "Train Loss", xlabel = "Epochs", ylabel = "Loss",
        title = title, minorgrid = true, xtickfontsize = 18,
        ytickfontsize = 18, legendfontsize = 18, xguidefontsize=18, 
        yguidefontsize=18, titlefontsize = 20, size = figsize)
    end
    
end


function plot_error_curve(trn_err, tst_err = nothing; title = "Train Error w.r.t. Epochs", figsize = (700, 500), epochs = nothing)
    if epochs == nothing
         epochs = 1:size(trn_err,1)
    end    
    if tst_err != nothing
        if size(trn_err,1) != size(tst_err,1)
            print("The loss vectors for train and test do not have the same size")
            return 0
        else
            plot(epochs,trn_err ,label = "Train Error", xlabel = "Epochs", ylabel = "Loss",
            title = title, minorgrid = true, xtickfontsize = 18,
            ytickfontsize = 18, legendfontsize = 18, xguidefontsize=18, 
            yguidefontsize=18, titlefontsize = 20, size = figsize)
            plot!(epochs,tst_err ,label = "Test Loss", title = title)
        end
    else
        plot(epochs,trn_err ,label = "Train Error", xlabel = "Epochs", ylabel = "Loss",
        title = title, minorgrid = true, xtickfontsize = 18,
        ytickfontsize = 18, legendfontsize = 18, xguidefontsize=18, 
        yguidefontsize=18, titlefontsize = 20, size = figsize)
    end
    
end


function plot_image_grid(im_tensor; grid_x_size = 3, grid_y_size = 3, title = "")
    if size(im_tensor, 4) < (grid_x_size * grid_y_size)
        println("Grid Size is bigger than number of input images. Adjust your tensor!")
        return nothing
    end
    im_size_x = size(im_tensor,1)
    im_size_y = size(im_tensor,2)
    empty_grid = (zeros(im_size_x * grid_x_size, im_size_y * grid_y_size, size(im_tensor,3)))

    for i in (1:grid_x_size)
        for j in (1:grid_y_size)
            empty_grid[(i-1)*im_size_x + 1:(i)*im_size_x, (j-1)*im_size_y + 1:(j)*im_size_y,:] .= im_tensor[:,:,:,i*j]
        end
    end

    if size(empty_grid,3) == 1
        plot(Matrix{Gray{Float32}}(reshape(empty_grid, (im_size_x * grid_x_size,im_size_y * grid_y_size))), title = title, size = (400,400),font =  "Courier", xtick = false, ytick = false)
    else
        println("RGB image plotting not implemented yet!")
    end
    
end


function plot_weight_histograms(theta)
    L = length(theta)
    for i = 1:L
        s = size(theta[i])
        display(histogram([flatten(Array{Float32}(value(theta[i])))...],title = "Size of Weight : $s", xlabel = "Values", ylabel = "count", label = "w", grid = true,nbins=100))
    end
end

end