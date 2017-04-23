module SAMPLER
using Knet

export increase_in_energy, sample_pool_group

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

function sample_pool_group(prob_hidden_one, prob_pool_zero)
sample_hidden_one = map(x-> rand() > x ? 1 : 0, prob_hidden_one)
sample_pool_zero = rand() > prob_pool_zero ? 1: 0
return (sample_hidden_one, sample_pool_zero)
end

end
