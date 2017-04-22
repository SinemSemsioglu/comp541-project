using Knet

function main(;
              mode=0, #will there be more than one modes
              cd=1, #contrastive divergence
              #batch=64, will this always be one image?
              filtersize=10, #assume square
              numfilters=24,
              sparsity=0.003,
              sparsity_lr=5,
              gradient_lr=0.1,
              pool=2, #C
              input=[], #how will this parameter be?
              winit = 0.001,
              #otype="Adam()",
              atype=(gpu()>=0 ? KnetArray{Float32} : Array{Float32}),
              model=initmodel(filtersize, numfilters),
              #state=initstate(model,batch), do I need a state?
              #optim=initoptim(model,otype), will I need opt.
              #dropout=0,
              )
    if mode == 0

    elseif mode == 1

    elseif mode == 2

    else
        error("mode=$mode")
    end

    train(input, model[1], model[2], model[3], pool, cd)
    return (model, state, optim)
end

#initialize weights
# one bias for input, c
# K: numfilters
# K length biases for hidden unit groups
# K filters of size filtersize x filtersize
# should they be stored as in cnns (filtersize, filtersize, numchannels in, numchannels out)
# only working with 1 channel rn.
# and one instance

function initmodel(filtersize, numfilters; winit=0.001)
  filters = winit * randn(Float64, filtersize, filtersize, 1, numfilters)
  hidden_bias = zeros(Float64, 1,1,numfilters,1)
  visible_bias = randn(Float64, 1,1,1,1); #check dimensions for This

  return (filters, hidden_bias, visible_bias)
end

function train(visible, filters, hidden_bias, visible_bias, pool_size, cd)
  hidden_post, hidden_sample, pool_post, pool_sample = sample_hidden(visible, filters, hidden_bias, pool_size)

  hidden_post_org = copy(hidden_post)
  visible_org = copy(visible)

  for c = 1:cd
    # check here if real of binary
    visible = sample_visible_real(hidden_sample, filters, visible_bias, size(visible,1))
    hidden_post, hidden_sample, pool_post, pool_sample = sample_hidden(visible, filters, hidden_bias, pool_size)
  end

  #update weights
  norm_d = 1/size(hidden_post,1)^2 # normalizing denominator

  g_loss = Array{Float64}(size(filters))
  for k=1:size(filters,3)
    h_org = reshape(hidden_post_org[:,:,k,1], size(hidden_post,1), size(hidden_post,2), 1, size(hidden_post,4))
    h = reshape(hidden_post[:,:,k,1], size(hidden_post,1), size(hidden_post,2), 1, size(hidden_post,4))

    g_loss[:,:,:,k] = norm_d * (conv4(h_org, visible_org; mode=1) - conv4(h, visible; mode=1))
  end


  b_sparsity_reg = sparsity - norm_d * sum(hidden_post, (1, 2, 4))
  b_loss = norm_d * sum(hidden_post_org - hidden_post, (1,2,4))
  c_loss = 1/size(visible,1)^2 * sum(visible_org - visible, (1,2,4))

  filters -= gradient_lr * g_loss
  hidden_bias -= gradient_lr * b_loss + sparsity_lr * b_sparsity_reg
  visible_bias -= c_loss
end

function sample_hidden(visible, filters, hidden_bias, pool_size)
  energies = increase_in_energy(visible, filters, hidden_bias)
  num_filters = size(energies,3)
  hidden_width = size(energies,1)
  hidden_height = size(energies,2)

  # width and height are not divisible by poolsize, padding?
  sample_width = Int(floor(hidden_width/pool_size))
  sample_height = Int(floor(hidden_width/pool_size))

  hidden_samples = Array{Float64}(hidden_width, hidden_height, num_filters, 1)
  pool_samples = Array{Float64}(sample_width, sample_height, num_filters, 1)
  hidden_posts = Array{Float64}(hidden_width, hidden_height, num_filters, 1)
  pool_posts = Array{Float64}(sample_width, sample_height, num_filters, 1)

  for k=1:size(energies,3)
    for i=1:sample_width
      for j=1:sample_height
        #check indices for energies
        h_width_indices = i * pool_size - 1:i * pool_size
        h_height_indices =  j * pool_size - 1:j*pool_size

        hidden_post, pool_post = posterior_pool_group(energies[h_width_indices,h_height_indices, k, :])
        hidden, pool = sample_pool_group(hidden_post, pool_post)

        hidden_posts[h_width_indices,h_height_indices,k, 1] = hidden_post
        pool_posts[i,j,k,1] = pool_post
        hidden_samples[h_width_indices,h_height_indices,k, 1] = hidden
        pool_samples[i,j,k,1] = pool
      end
    end
  end

  return (hidden_posts, hidden_samples, pool_posts, pool_samples)
end



function sample_visible_binary(hidden, filters, visible_bias, visible_dim)
    conv_sum = get_conv_sum(hidden, filters, visible_dim)
    return sigm(conv_sum .+ visible_bias)
end

function get_conv_sum(hidden, filters, visible_dim)
    num_groups = size(hidden, 3)
    #assume square all arrays are square (hidden and filters here but the same assumption implicitly applies to the visible -> input)
    out_dim = floor(size(hidden,1) - size(filters,1)) + 1
    conv_sum = zeros(visible_dim, visible_dim, 1, size(hidden,4))
    padd = Int((visible_dim - out_dim) / 2)

    for group=1:num_groups
        h = reshape(hidden[:,:,group,:], size(hidden, 1), size(hidden, 2), 1, size(hidden, 4))
        w = reshape(filters[:,:,:,group], size(filters, 1), size(filters, 2), size(filters,3), 1)
        conv_sum += conv4(w, h; padding = padd)
    end

    return conv_sum
end

function sample_visible_real(hidden, filters, visible_bias, visible_dim)
    conv_sum = get_conv_sum(hidden, filters, visible_dim)
    return conv_sum .+ visible_bias
end

function increase_in_energy(visible, filters, hidden_bias)
  d = size(visible, 1) - size(filters,1) + 1
  e = Array{Float64}(d,d,size(filters,4),size(visible,4))
  f = Array{Float64}(size(filters,1), size(filters,2), size(filters,3),1)
  for i = 1:size(filters,4)
      f[:,:,:] = filters[:,:,:,i]
      e[:,:,i,:] = conv4(f, visible, mode = 1) .+ hidden_bias[1,1,i,1]
  end
#return conv4(filters, visible; mode=1) .+ hidden_bias
  return e
end

# calculates p(pool unit = 0|v) p(hidden unit = 1|v)
function posterior_pool_group(energies)
# normalize log
  exp_energies = exp(energies)
  sum_exp_energies = sum(exp_energies) # sum along both dimensions

  prob_hidden_one = exp_energies ./ sum_exp_energies
  prob_pool_zero = 1/ (1 + sum_exp_energies)


  return (prob_hidden_one, prob_pool_zero)
end

function sample_pool_group(prob_hidden_one, prob_pool_zero)
    sample_hidden_one = map(x-> rand() > x ? 1 : 0, prob_hidden_one)
    sample_pool_zero = rand() > prob_pool_zero ? 1: 0
    return (sample_hidden_one, sample_pool_zero)
end
