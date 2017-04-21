using Knet

function main(;
              mode=0, #will there be more than one modes
              cd=1, #contrastive divergence
              #batch=64, will this always be one image?
              filtersize=10, #assume square
              numfilters=24,
              sparsity=0.003,
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
  hidden = sample_hidden(visible, filters, hidden_bias, pool_size)

  for c = 1:cd
    # check here if real of binary
    visible = sample_visible_real(hidden, filters, visible_bias)
    hidden = sample_hidden(visible, filters, hidden_bias, pool_size)
  end

  #update weights
end

function sample_hidden(visible, filters, hidden_bias, pool_size)
  energies = increase_in_energy(visible, filters, hidden_bias)
  num_filters = size(energies,3)
  hidden_width = size(energies,1)
  hidden_height = size(energies,2)

  # width and height are not divisible by poolsize, padding?
  sample_width = Int(floor(hidden_width/pool_size))
  sample_height = Int(floor(hidden_width/pool_size))
  samples = Array{Float64}(sample_width, sample_height, num_filters, 1)

  for k=1:size(energies,3)
    for i=1:sample_width
      for j=1:sample_height
        #check indices for energies
        samples[i,j,k,1] = sample_pool_group(energies[i * pool_size - 1:i * pool_size, j * pool_size - 1:j*pool_size, k, :])
      end
    end
  end

  return samples
end



function sample_visible_binary(hidden, filters, visible_bias)
    conv_sum = get_conv_sum(hidden, filters)
    return sigm(conv_sum .+ visible_bias)
end

function get_conv_sum(hidden, filters)
    num_groups = size(hidden, 3)
    #assume square all arrays are square (hidden and filters here but the same assumption implicitly applies to the input)
    out_dim = floor(size(hidden,1) - size(filters,1)) + 1
    conv_sum = zeros(out_dim, out_dim, 1, size(hidden,4))

    for group=1:num_groups
        h = reshape(hidden[:,:,group,:], size(hidden, 1), size(hidden, 2), 1, size(hidden, 4))
        w = reshape(filters[:,:,:,group], size(filters, 1), size(filters, 2), size(filters,3), 1)
        conv_sum += conv4(w, h)
    end

    return conv_sum
end

function sample_visible_real(hidden, filters, visible_bias)
    conv_sum = get_conv_sum(hidden, filters)
    return conv_sum .+ visible_bias
end

function increase_in_energy(visible, filters, hidden_bias)
  d = size(visible, 1) - size(filters,1) + 1
  e = Array{Float64}(d,d,size(filters,4),size(visible,4))
  for i = 1:size(filters,4)
      e[:,:,i,:] = conv4(filters[:,:,:,i], visible, mode = 1) .+ hidden_bias[1,1,k,1]
  end
#return conv4(filters, visible; mode=1) .+ hidden_bias
  return e
end

# calculates p(hidden unit = 0|v) p(hidden unit = 1|v)
# and checks which one is greater
function sample_pool_group(energies)
  exp_energies = exp(energies)
  sum_exp_energies = sum(exp_energies) # sum along both dimensions
  h_s = map(x->x/(1+sum_exp_energies), exp_energies)
  print(h_s[1,1], " ", h_s[1,2], " ",h_s[2,1], " ",h_s[2,2])
# pred = map(x-> x/(1 + sum_exp_energies) > 1/(1 + sum_exp_energies), exp_energies)
#  return sum(pred) >= 1
  zero_prob = 1/ (1 + sum_exp_energies)
  print(": ", zero_prob, "\n")
  return zero_prob < 0.5 ? 1 : 0
end
