module SAMPLER
using Knet, Distributions

export increase_in_energy_hidden, increase_in_energy_pool, sample_pool_group, sample_visible_binary, sample_visible_real, calculate_posterior, find_nan_and_replace, find_inf_and_replace

function sample_visible_binary(hidden, filters, visible_bias, visible_dims)
    conv_sum = get_conv_sum(hidden, filters, visible_dims)
    prob = sigm(conv_sum .+ visible_bias)
    return float(map(x -> find(rand(Multinomial(1, [1-x, x]),1))[1] - 1 , prob))
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
    means = conv_sum .+ visible_bias
    samples = map(x-> rand(Normal(x,1),1)[1], means)
    return samples
end

function increase_in_energy_hidden(visible, filters, hidden_bias)
    d = size(visible, 1) - size(filters,1) + 1
    e = zeros(d,d,size(filters,4),size(visible,4))
    f = zeros(size(filters,1), size(filters,2), size(filters,3),1)
    for i = 1:size(filters,4)
        f[:,:,:] = filters[:,:,:,i]
        e[:,:,i,:] = conv4(f, visible, mode = 1) .+ hidden_bias[1,1,i,1]
    end
    #return conv4(filters, visible; mode=1) .+ hidden_bias
    return e
end

function sample_pool_group(prob_hidden_one, prob_pool_zero)
#sample_hidden = map(x-> rand() > x ? 1 : 0, prob_hidden_one)
    sample_hidden = map(x -> find(rand(Multinomial(1, [1-x, x]),1))[1] - 1, prob_hidden_one)
    one_indices = find(sample_hidden)

    # only 1 of them should be zero
    if (size(one_indices,1) > 1 )
        # hidden variable to keep as one
        rand_one_index = rand(1:size(one_indices,1))
        rest_ones = [one_indices[1: (rand_one_index - 1)] ; one_indices[(rand_one_index + 1): end]]

        for rest_one in rest_ones
            row = mod(rest_one - 1 , size(sample_hidden, 1)) + 1
            col = div(rest_one - 1, size(sample_hidden,2)) +1
            sample_hidden[row,col] = float(0)
        end
    end

    # if I am sampling by looking at the hidden units, then I don't need the prob distro?
    sample_pool = sum(sample_hidden)
    return (sample_hidden, sample_pool)
end



#  posterior pool group is a specific case of this where top_down is 0 call this func. instead
function calculate_posterior(bottom_up, top_down)
    total = bottom_up .+ top_down
    total = find_inf_and_replace(total, log(realmax(Float64)), log(realmin(Float64)))
    total = total .- maximum(total,1)

    exp_energies = exp(total)
    sum_exp_energies = sum(exp_energies)

    # handling 0 division
    if sum_exp_energies == -1
    # this doesn't really make sense
        prob_hidden_one = zeros(size(bottom_up))
        prob_pool_zero = 1
    else
        prob_hidden_one = exp_energies/(1+ sum_exp_energies)
        prob_pool_zero = 1/(1 + sum_exp_energies)
    end

    # check inf and nan
    prob_hidden_one = find_inf_and_replace(prob_hidden_one, 1, 0)
    prob_hidden_one = find_nan_and_replace(prob_hidden_one, 0)

    if prob_pool_zero == - Inf || isnan(prob_pool_zero)
        prob_pool_zero = 0
    elseif prob_pool_zero == Inf
        prob_pool_zero = 1
    end

    return (prob_hidden_one, prob_pool_zero)
end

function increase_in_energy_pool(weight, hidden, pool_dims)
    num_groups = size(weight,3)
    num_channels = size(hidden,3)
    energies = zeros(pool_dims)

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

function find_inf_and_replace(matrix, r_max, r_min)
    inf_indices = find(function(x) return x == Inf end, matrix)
    n_inf_indices = find(function(x) return x == -Inf end, matrix)

    i_row_indices = mod(inf_indices - 1 , size(matrix, 1)) + 1
    i_col_indices = div(inf_indices - 1, size(matrix,2)) +1

    n_row_indices = mod(n_inf_indices - 1 , size(matrix, 1)) + 1
    n_col_indices = div(n_inf_indices - 1, size(matrix,2)) +1

    for i in inf_indices
        matrix[i_row_indices[i], i_col_indices[i]] = r_max
    end

    for i in n_inf_indices
        prob_hidden_one[n_row_indices[i], n_col_indices[i]] = r_min
    end

    return matrix
end


function find_nan_and_replace(matrix, r)
    nan_indices = find(isnan, matrix)

    row_indices = mod(nan_indices - 1 , size(matrix, 1)) + 1
    col_indices = div(nan_indices - 1, size(matrix,2)) +1

    for i in nan_indices
        matrix[row_indices[i], col_indices[i]] = r
    end

    return matrix
end

end
