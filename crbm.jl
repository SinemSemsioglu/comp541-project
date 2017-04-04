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

function initmodel(filtersize, numfilters; winit=0.001)
  filters = winit * randn(Float32, filtersize, filtersize, 1, numfilters)
  hidden_bias = zeros(Float32, 1,1,numfilters,1)
  visible_bias = randn(Float32, 1,1,1,1); #check dimensions for This

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
  samples = Array{Float32}(floor(hidden_width/pool_size), floor(hidden_height/pool_size), num_filters)

  for k=1:size(energies,3)
    for i=1:pool_size:size(energies,1)
      for j=1:pool_size:size(energies,2)
        samples[i,j,k] = sample_pool_group(energies[i:i+pool_size-1,j:j+pool_size-1])
      end
    end
  end

end



function sample_visible_binary(hidden, filters, visible_bias)
    return sigm(sum(conv4(filters, hidden),3) .+ visible_bias)
end

function sample_visible_real(hidden, filters, visible_bias)
    return sum(conv4(filters, hidden),3) .+ visible_bias
end

function increase_in_energy(visible, filters, hidden_bias)
  return conv4(filters, visible; mode=1) .+ hidden_bias
end

# calculates p(hidden unit = 0|v) p(hidden unit = 1|v)
# and checks which one is greater
function sample_pool_group(energies)
  exp_energies = exp(energies)
  sum_exp_energies = sum(exp_energies) # sum along both dimensions
  pred = map(x-> x/(1 + sum_exp_energies) > 1/(1 + sum_exp_energies), exp_energies)
  return sum(pred) >= 1
end
