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
                sparsity=[0.003,0.005],
                gradient_lr=0.1,
                sparsity_lr=5,
                cd=1,
                batch=1,
                mode=0 #input is real for mode 0, binary for mode 1
                )

    models = Any[]
    states = Any[]

    for layer = 1:numlayers
        if(layer > 1)
            mode = 1
            prev_pool = states[layer-1][3]
            model,state = CRBM.main(;mode=mode,cd_=cd, filtersize_=filtersize[layer],  numfilters_=numfilters[layer],
                                sparsity_=sparsity[layer], gradient_lr_=gradient_lr, sparsity_lr_=sparsity_lr,
                                cd_=cd, pool_=pool[layer], input=prev_pool)
        else
            model,state = CRBM.main(;mode=mode,cd_=cd, filtersize_=filtersize[layer], numfilters_=numfilters[layer],
                           sparsity_=sparsity[layer], gradient_lr_=gradient_lr, sparsity_lr_=sparsity_lr,
                           cd_=cd, pool_=pool[layer],input=input)


        end

        push!(models, model)
        push!(states, state)
    end

    # for efficiency the energies calculated during training can also be returned
    # this can be taken into the previous loop
    poolz = Any[]

    for layer=1:numlayers
        bottom_visible = states[layer][1]
        bottom_hidden = states[layer][2]
        bottom_pool = states[layer][3]
        bottom_weights = models[layer]

        bottom_up = SAMPLER.increase_in_energy(bottom_visible, bottom_weights[1], bottom_weights[2])
        pool_size = pool[layer]

        # there will be no top layer if this is the last one
        if (layer != numlayers)
            top_hidden_weights = models[layer + 1]
            top_hidden_values = states[layer+1][2]
            top_down = increase_in_energy_pool(top_hidden_weights[1], top_hidden_values, size(bottom_pool))
        else
            top_down = zeros(size(bottom_pool))
        end

        # width and height are not divisible by poolsize, padding?
        sample_width = Int(floor(size(bottom_hidden,1)/pool_size))
        sample_height = Int(floor(size(bottom_hidden,2)/pool_size))
        num_filters = size(bottom_hidden,3)

        # 1's below should be replaced by batch size
        # hidden_samples = Array{Float64}(hidden_width, hidden_height, num_filters, 1)
        pool_samples = Array{Float64}(sample_width, sample_height, num_filters, 1)
        #hidden_posts = Array{Float64}(hidden_width, hidden_height, num_filters, 1)
        #pool_posts = Array{Float64}(sample_width, sample_height, num_filters, 1)

        for k=1:size(bottom_hidden, 3)
            for i=1:sample_width
                for j=1:sample_height
                    h_width_indices = i * pool_size - 1:i * pool_size
                    h_height_indices =  j * pool_size - 1:j*pool_size

                    hidden_post, pool_post = calculate_posterior(bottom_up[h_width_indices, h_height_indices, k, :], top_down[i,j,k,:])
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

#  posterior pool group is a specific case of this where top_down is 0 call this func. instead
function calculate_posterior(bottom_up, top_down)
    exp_energies = exp(bottom_up .+ top_down)
    sum_exp_energies = sum(exp_energies)

    prob_hidden_one = exp_energies/(1+ sum_exp_energies)
    prob_pool_zero = 1/(1 + sum_exp_energies)

    return (prob_hidden_one, prob_pool_zero)
end

function increase_in_energy_pool(weight, hidden, pool_dims)
    num_groups = size(weight,3)
    num_channels = size(hidden,3)
    energies = Array{Float64}(pool_dims)

    for k=1:num_groups
        out_dim = size(hidden,1) - size(weight,1) + 1 # assume square
        padd = Int((pool_dims[1] - out_dim) / 2) # to have the sime size as the pooling layer

        for c=1:num_channels
            w = reshape(weight[:,:,k,c], size(weight,1), size(weight,2),1,1)
            h = reshape(hidden[:,:,c,:], size(hidden,1), size(hidden,2), 1, size(hidden,4))
            energies[:,:,k,:] += conv4(w, h; padding = padd)
        end
    end

    return energies
end

end
