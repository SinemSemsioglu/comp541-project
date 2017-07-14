module CDBN

export main

import CRBM
import SAMPLER
using Knet

function main(;
                input=rand(100,100,1,1),
                numlayers=2,
                filtersize=[10,10],
                numfilters=[24,100],
                pool=[2,2],
                modelz=[],
                sparsity=[0.003,0.005],
                gradient_lr=0.1,
                sparsity_lr=5,
                cd=1,
                batch=1,
                return_mode=0, # 0 for training, 1 for activations
                mode=0 #input is real for mode 0, binary for mode 1
                )

#print("cdbn filtersizes: ", filtersize, "\n");
    if return_mode == 0
        return train(input, numlayers, filtersize, numfilters, pool, modelz, sparsity, gradient_lr, sparsity_lr, cd, batch, mode, return_mode)
    elseif return_mode == 1
        return get_pool_activations(input, modelz, numlayers, pool, return_mode)
    end
end

function train(input, numlayers, filtersize, numfilters, pool, modelz, sparsity, gradient_lr, sparsity_lr, cd, batch, mode, return_mode)
    models = Any[]
    states = Any[]

    for layer = 1:numlayers
        if(layer > 1)
            mode = 1
            prev_pool = states[layer-1][3]
            if(size(modelz,1) != 0)
                model,state = CRBM.main(;mode=mode,return_mode=return_mode, cd_=cd, filtersize_=filtersize[layer],  numfilters_=numfilters[layer], sparsity_=sparsity[layer], gradient_lr_=gradient_lr, sparsity_lr_=sparsity_lr, pool_=pool[layer], input=prev_pool, model=modelz[layer])
            else
                model,state = CRBM.main(;mode=mode, return_mode = return_mode, cd_=cd, filtersize_=filtersize[layer], numfilters_=numfilters[layer], sparsity_=sparsity[layer], gradient_lr_=gradient_lr, sparsity_lr_=sparsity_lr, pool_=pool[layer], input=prev_pool)
            end
        else
            if(size(modelz,1) != 0)
                 model,state = CRBM.main(;mode=mode, return_mode = return_mode, cd_=cd, filtersize_=filtersize[layer], numfilters_=numfilters[layer], sparsity_=sparsity[layer], gradient_lr_=gradient_lr, sparsity_lr_=sparsity_lr, pool_=pool[layer],input=input,model=modelz[layer])
            else
                model,state = CRBM.main(;mode=mode, return_mode = return_mode, cd_=cd, filtersize_=filtersize[layer], numfilters_=numfilters[layer], sparsity_=sparsity[layer], gradient_lr_=gradient_lr, sparsity_lr_=sparsity_lr, pool_=pool[layer],input=input)
            end
        end

        push!(models, model)
        push!(states, state)
    end

    return models, states
end

function get_pool_activations(input, modelz, numlayers, pool, return_mode)
    states = get_image_states(input, modelz, numlayers, pool, return_mode)
    pools = calculate_pool_activations(modelz, states, numlayers, pool)
    return pools
end

function get_image_states(input, modelz, numlayers, pool, return_mode)
    states = Any[]

    for layer = 1:numlayers
        if(layer > 1)
            mode = 1
            prev_pool = states[layer-1][3]
            model,state = CRBM.main(; return_mode = return_mode, pool_=pool[layer], input=prev_pool, model=modelz[layer])
        else
            model,state = CRBM.main(; return_mode = return_mode, pool_=pool[layer],input=input,model=modelz[layer])
        end

        push!(states, state)
    end

    return states
end

function calculate_pool_activations(models, states, numlayers, pool)

    # for efficiency the energies calculated during training can also be returned
    # this can be taken into the previous loop
    poolz = Any[]

    for layer=1:numlayers
        bottom_visible = states[layer][1]
        bottom_hidden = states[layer][2]
        bottom_pool = states[layer][3]
        bottom_weights = models[layer]

        bottom_up = SAMPLER.increase_in_energy_hidden(bottom_visible, bottom_weights[1], bottom_weights[2])
        pool_size = pool[layer]

        # there will be no top layer if this is the last one
        if (layer != numlayers)
            top_hidden_weights = models[layer + 1]
            top_hidden_values = states[layer+1][2]
            top_down = SAMPLER.increase_in_energy_pool(top_hidden_weights[1], top_hidden_values, size(bottom_pool))
        else
            top_down = zeros(size(bottom_pool))
        end

        # width and height are not divisible by poolsize, padding?
        sample_width = Int(floor(size(bottom_hidden,1)/pool_size))
        sample_height = Int(floor(size(bottom_hidden,2)/pool_size))
        num_filters = size(bottom_hidden,3)

        # 1's below should be replaced by batch size
        # hidden_samples = zeros(hidden_width, hidden_height, num_filters, 1)
        pool_samples = zeros(sample_width, sample_height, num_filters, 1)
        #hidden_posts = zeros(hidden_width, hidden_height, num_filters, 1)
        #pool_posts = zeros(sample_width, sample_height, num_filters, 1)

        for k=1:size(bottom_hidden, 3)
            for i=1:sample_width
                for j=1:sample_height
                    h_width_indices = i * pool_size - 1:i * pool_size
                    h_height_indices =  j * pool_size - 1:j*pool_size

                    hidden_post, pool_post = SAMPLER.calculate_posterior(bottom_up[h_width_indices, h_height_indices, k, :], top_down[i,j,k,:])
                    hidden_sample, pool_sample = SAMPLER.sample_pool_group(hidden_post, pool_post)

                    #  hidden_posts[h_width_indices,h_height_indices,k, 1] = hidden_post
                    #pool_posts[i,j,k,1] = pool_post
                    #hidden_samples[h_width_indices,h_height_indices,k, 1] = hidden_sample
                    pool_samples[i,j,k,1] = pool_sample
                end
            end
        end

        push!(poolz, pool_samples)
    end

    return poolz
end

end
