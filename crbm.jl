module CRBM
using Knet
import SAMPLER

export main

function main(;
              mode=0, #will there be more than one modes, maybe for real binary visible
              return_mode = 0, # 0 for training, 1 for activations
              cd_=1, #contrastive divergence
                #batch=1, will this always be one image?
              filtersize_=10, #assume square
              numfilters_=24,
              sparsity_=0.003,
              sparsity_lr_=5,
              gradient_lr_=0.1,
              pool_=2, #C
              input=rand(100,100,1,1), #how will this parameter be?
              winit = 0.001,
              #otype="Adam()",
              atype_=(gpu()>=0 ? KnetArray{Float64} : Array{Float64}),
              model=initmodel(input, filtersize_, numfilters_),
              #state=initstate(model,batch), do I need a state?
              #optim=initoptim(model,otype), will I need opt.
              )
    if mode == 0
        real_input = true
    elseif mode == 1
        real_input = false
    else
        error("mode=$mode")
    end

    global sparsity = sparsity_
    global sparsity_lr = sparsity_lr_
    global gradient_lr = gradient_lr_
    global pool_size = pool_
    global numfilters = numfilters_
    global filtersize = filtersize_
    global cd = cd_
    global atype = atype_

#print(" return mode: ", return_mode, " mode: ", mode);

    if return_mode == 0
        model, state = train(input, model[1], model[2], model[3], real_input)
    elseif return_mode == 1
        model,state = get_activations(input, model[1], model[2], model[3])
    end

    #also return optim if used
    return (model, state)
end

# in order to be used for SVM this will calculate the pool activations without training
function get_activations(visible, filters, hidden_bias, visible_bias)
    hidden_post, hidden_sample, pool_post, pool_sample = sample_hidden(visible, filters, hidden_bias)

    return ([filters, hidden_bias, visible_bias], [visible, hidden_sample, pool_sample])
end

#initialize weights
# one bias for input, c
# K: numfilters
# K length biases for hidden unit groups
# K filters of size filtersize x filtersize
# should they be stored as in cnns (filtersize, filtersize, numchannels in, numchannels out)
# only working with 1 channel rn.
# and one instance

function initmodel(input, filtersize, numfilters; winit=0.001)
  num_channels = size(input,3)
  filters = winit * randn(Float64, filtersize, filtersize, num_channels, numfilters)
  hidden_bias = zeros(Float64, 1,1,numfilters,1)
  visible_bias = zeros(Float64, 1,1,num_channels,1); #check dimensions for This

  return (filters, hidden_bias, visible_bias)
end

function train(visible, filters, hidden_bias, visible_bias, real_input)
#print(" gradient lr: ", gradient_lr, " numfilters: ", numfilters, " filtersize: ", filtersize, " pool: ", pool_size, "\n");

hidden_post, hidden_sample, pool_post, pool_sample = sample_hidden(visible, filters, hidden_bias)

  hidden_post_org = copy(hidden_post)
  visible_org = copy(visible)

  for c = 1:cd
    # check here if real of binary
    if (real_input == true)
        visible = SAMPLER.sample_visible_real(hidden_sample, filters, visible_bias, size(visible))
    else
        visible = SAMPLER.sample_visible_binary(hidden_sample, filters, visible_bias, size(visible))
    end

    hidden_post, hidden_sample, pool_post, pool_sample = sample_hidden(visible, filters, hidden_bias)
  end

  #update weights
  norm_d = 1/size(hidden_post,1)^2 # normalizing denominator

  g_loss = zeros(size(filters))

  for c=1:size(visible,3)
    for k=1:size(filters,4)
        h_org = reshape(hidden_post_org[:,:,k,:], size(hidden_post,1), size(hidden_post,2), 1, size(hidden_post,4))
        h = reshape(hidden_post[:,:,k,:], size(hidden_post,1), size(hidden_post,2), 1, size(hidden_post,4))

        v_org = reshape(visible_org[:,:,c,:], size(visible,1), size(visible,2), 1, size(visible,4))
        v = reshape(visible[:,:,c,:], size(visible,1), size(visible,2), 1, size(visible,4))

        losses = norm_d * (conv4(h_org, v_org; mode=1) - conv4(h, v; mode=1))
        g_loss[:,:,c,k] = losses
    end
  end

#print("reconstruction error: ", mean((visible_org - visible).^2), "\n")
# print("sparsity: ", sum(hidden_sample)/ (size(hidden_sample,1)* size(hidden_sample,2)*size(hidden_sample,3)*size(hidden_sample,4)), "\n")


  b_sparsity_reg = sparsity - norm_d * sum(hidden_post, (1, 2, 4))
  b_loss = norm_d * sum(hidden_post_org - hidden_post, (1,2,4))
  c_loss = (1/size(visible,1)^2) * sum(visible_org - visible, (1,2,4))

  filters += gradient_lr * (g_loss .- 0.01 * abs(filters) - 0 * (map(x-> x > 0 ? 1:0, filters)*2-1))
  hidden_bias += gradient_lr * (b_loss + sparsity_lr * b_sparsity_reg)
  visible_bias += c_loss

  # check if model is updated
  return ([filters, hidden_bias, visible_bias], [visible, hidden_sample, pool_sample])
end

function sample_hidden(visible, filters, hidden_bias)
  energies = SAMPLER.increase_in_energy_hidden(visible, filters, hidden_bias)
  num_filters = size(energies,3)
  hidden_width = size(energies,1)
  hidden_height = size(energies,2)

  # width and height are not divisible by poolsize, padding?
  sample_width = Int(floor(hidden_width/pool_size))
  sample_height = Int(floor(hidden_width/pool_size))

# 1's below should be replaced by batch size
  hidden_samples = zeros(hidden_width, hidden_height, num_filters, 1)
  pool_samples = zeros(sample_width, sample_height, num_filters, 1)
  hidden_posts = zeros(hidden_width, hidden_height, num_filters, 1)
  pool_posts = zeros(sample_width, sample_height, num_filters, 1)

  for k=1:size(energies,3)
    for i=1:sample_width
      for j=1:sample_height
        #check indices for energies
        h_width_indices = i * pool_size - 1:i * pool_size
        h_height_indices =  j * pool_size - 1:j*pool_size

        hidden_post, pool_post = SAMPLER.calculate_posterior(energies[h_width_indices,h_height_indices, k, :], 0)
        hidden, pool = SAMPLER.sample_pool_group(hidden_post, pool_post)

        hidden_posts[h_width_indices,h_height_indices,k, 1] = hidden_post
        pool_posts[i,j,k,1] = pool_post
        hidden_samples[h_width_indices,h_height_indices,k, 1] = hidden
        pool_samples[i,j,k,1] = pool
      end
    end
  end

  return (hidden_posts, hidden_samples, pool_posts, pool_samples)
end





end
