module TRAIN
using Knet, JLD
import CDBN
import CRBM

export main

function main(;
                images=rand(100,100,1,100), # mock data of 100 instances of 100x100 images
                initial_model=[],
                numlayers=2,
                filtersize=[10,10],
                numfilters=[24,100],
                pool=[2,2],
                sparsity=[0.003,0.005],
                gradient_lr=0.1,
                sparsity_lr=5,
                max_iterations=[100,100],
                debug=0,
                cd=1,
                batch=1,
                save_model_data=0,
                model_path="model.jld",
                mode=0 #input is real for mode 0, binary for mode 1
                )

    crbm_models = Any[]
    crbm_states = Any[]
    crbm_hiddens = Any[]
    crbm_recons = Any[]
    initial_model_l = []
    next_input  = []
    stats = Any[]
    progress = Any[]

    for layer=1:numlayers
        if debug == 1
            print("layer: ", layer, "\n")
        end

        input = []

        if layer == 1
            mode = 0
            input = images
        else
            mode = 1
            input = crbm_recons[layer-1]
        end

        if size(initial_model,1) > 0 initial_model_l = initial_model[layer] end

        models, recons, model, state, stat = train_crbm(input, mode, initial_model_l, filtersize[layer], numfilters[layer], pool[layer], sparsity[layer], gradient_lr, sparsity_lr, cd, max_iterations[layer], debug)
        hidden, next_input = get_hidden_layers(input, model, pool[layer], mode, size(state[2]), size(state[3]))


        push!(stats, stat)
        push!(crbm_hiddens, hidden)
        push!(crbm_recons, next_input)
        push!(crbm_models, model)
        push!(crbm_states, state)
        push!(progress, (models, recons))

        if save_model_data == 1
            layer_path = string("layer_", layer, "_", model_path)
            save(layer_path, "progress", progress, "model",model, "state", state, "recons", next_input, "stats", )
        end
    end

    return (crbm_models, crbm_states, crbm_hiddens, crbm_recons, stats)
end

function get_hidden_layers(images, model, pool, mode, hidden_size, pool_size)
    num_images = size(images,4)

    pools = zeros(pool_size[1],pool_size[2], pool_size[3], num_images)
    hiddens = zeros(hidden_size[1],hidden_size[2], hidden_size[3], num_images)
    activation_mode = 1

    for imIndex=1:num_images
        image = images[:,:,:,imIndex]
        image = reshape(image, size(image,1), size(image,2), size(image,3),1)
        model, state = CRBM.main(return_mode=activation_mode, input=image, pool_=pool, mode=mode, model=model)
        hiddens[:,:,:,imIndex] = state[2]
        pools[:,:,:,imIndex] = state[3]
    end

    return hiddens, pools
end

function train_crbm(images, mode, initial_model,filtersize, numfilters, pool, sparsity, gradient_lr, sparsity_lr, cd, max_iterations, debug)
    cdbn_mode = 0
    prev_model = []
    prev_state = []
    optim = []

    recon_errs = []
    sparsity_rates = []
    grad_diffs = []
    models = Any[]
    recons = Any[]

    numImages = size(images,4)
    train_mode = 0

    for epoch=0:max_iterations
        imIndex = mod(epoch, numImages) + 1
        image = images[:,:,:,imIndex]
        image = reshape(image, size(image,1), size(image,2), size(image,3),1)

        # train
        if epoch > 0
            model, state, optim,recon_err, sparsity_rate = CRBM.main(return_mode=train_mode, input=image, numfilters_=numfilters, filtersize_=filtersize, pool_=pool, sparsity_=sparsity, gradient_lr_=gradient_lr, sparsity_lr_=sparsity_lr, cd_=cd, mode=mode, model=deepcopy(prev_model), optim=optim)
        else
            optim = [Adam(;lr=gradient_lr,beta2=0), Adam(;lr=gradient_lr,beta2=0), Adam(;lr=gradient_lr,beta2=0)]

            if size(initial_model,1) >0
                model, state, optim, recon_err, sparsity_rate = CRBM.main(return_mode=train_mode, input=image, numfilters_=numfilters,  filtersize_=filtersize, pool_=pool, sparsity_=sparsity, gradient_lr_=gradient_lr, sparsity_lr_=sparsity_lr, cd_=cd, model=initial_model, mode=mode, optim=optim)
            else
                model, state, optim, recon_err, sparsity_rate = CRBM.main(return_mode=train_mode, input=image, numfilters_=numfilters,  filtersize_=filtersize, pool_=pool, sparsity_=sparsity, gradient_lr_=gradient_lr, sparsity_lr_=sparsity_lr, cd_=cd, mode=mode, optim=optim)
            end
        end

        if (epoch % 20 == 0)
          push!(models, model)
          push!(recons, state[3])
        end

        if epoch > 0
            # check convergence
            ave_grad_diff = mean(prev_model[1][1] .- model[1][1])
            push!(grad_diffs, ave_grad_diff)

            if isapprox(prev_model[1], model[1])
                if debug == 1
                    print("converged on epoch ", epoch)
                end
                prev_model = model
                prev_state = state
                break
            elseif debug == 1 && epoch % 10 == 0
                print("epoch: ", epoch, " ave diff : ", ave_grad_diff, " sparsity: ", sparsity_rate, " recon err", recon_err, "\n")
            end
        end

        prev_state = state
        prev_model = model

        push!(recon_errs, recon_err)
        push!(sparsity_rates, sparsity_rate)
    end

    stat = (sparsity_rates, recon_errs, grad_diffs)
    return (models, recons, prev_model, prev_state, stat)
end

function train_full_cdbn(images, initial_model, numlayers, filtersize, numfilters, pool, sparsity, gradient_lr, sparsity_lr, cd, max_iterations, debug)
    cdbn_mode = 0
    prev_modelz = []
    prev_statez = []

    numImages = size(images,4)

    for imIndex=1:numImages
        image = images[:,:,:,imIndex]
        image = reshape(image, size(image,1), size(image,2), size(image,3),1)
        image -= mean(image)

        if imIndex > 1
            prev_modelz, prev_statez = CDBN.main(return_mode=cdbn_mode, input=image, numlayers=numlayers, numfilters=numfilters, filtersize=filtersize, pool=pool, sparsity=sparsity, gradient_lr=gradient_lr, sparsity_lr=sparsity_lr, cd=cd, batch=1, mode=0, modelz=prev_modelz)
        elseif size(initial_model,1) >0
            prev_modelz, prev_statez = CDBN.main(return_mode=cdbn_mode, input=image, numlayers=numlayers, numfilters=numfilters,  filtersize=filtersize, pool=pool, sparsity=sparsity, gradient_lr=gradient_lr, sparsity_lr=sparsity_lr, cd=cd, batch=1, modelz=initial_model, mode=0)
        else
            prev_modelz, prev_statez = CDBN.main(return_mode=cdbn_mode, input=image, numlayers=numlayers, numfilters=numfilters,  filtersize=filtersize, pool=pool, sparsity=sparsity, gradient_lr=gradient_lr, sparsity_lr=sparsity_lr, cd=cd, batch=1, mode=0)
        end
        # gradient_lr = gradient_lr / (1 + 0.1)

    end

end

end
