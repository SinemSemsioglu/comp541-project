module SAMPLER
using Knet

export increase_in_energy_hidden, increase_in_energy_pool, sample_pool_group, sample_visible_binary, sample_visible_real, calculate_posterior

function sample_visible_binary(hidden, filters, visible_bias, visible_dims)
conv_sum = get_conv_sum(hidden, filters, visible_dims)
return sigm(conv_sum .+ visible_bias)
end

function get_conv_sum(hidden, filters, visible_dims)
num_groups = size(hidden, 3)
num_channels = visible_dims[3]

#assume square all arrays are square (hidden and filters here but the same assumption implicitly applies to the visible -> input)
out_dim = floor(size(hidden,1) - size(filters,1)) + 1
conv_sum = zeros(visible_dims[1], visible_dims[2], visible_dims[3], visible_dims[4])

# to keep the visible size the same
padd = Int((visible_dims[1] - out_dim) / 2)

# check if looping through channels makes sense
for channel=1:num_channels
for group=1:num_groups
h = reshape(hidden[:,:,group,:], size(hidden, 1), size(hidden, 2), 1, size(hidden, 4))
w = reshape(filters[:,:,channel,group], size(filters, 1), size(filters, 2), 1, 1)
conv_sum[:,:,channel,:] += conv4(w, h; padding = padd)
end
end

return conv_sum
end

function sample_visible_real(hidden, filters, visible_bias, visible_dims)
conv_sum = get_conv_sum(hidden, filters, visible_dims)
return conv_sum .+ visible_bias
end

function increase_in_energy_hidden(visible, filters, hidden_bias)
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

function sample_pool_group(prob_hidden_one, prob_pool_zero)
sample_hidden_one = map(x-> rand() > x ? 1 : 0, prob_hidden_one)
sample_pool_zero = rand() > prob_pool_zero ? 1: 0
return (sample_hidden_one, sample_pool_zero)
end



#  posterior pool group is a specific case of this where top_down is 0 call this func. instead
function calculate_posterior(bottom_up, top_down)
bottom_up = bottom_up .- maximum(bottom_up,1)
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
