module CRBM
using Knet

export init

#assume square filters
function init(num_hidden, num_filters, num_input_channels; learning_rate = 0.1, winit = 0.1, target_sparsity=0.003, sparsity_lr=4)
	weights = Any[]
	push!(weights, winit * randn(num_hidden, num_hidden, num_input_channels, num_filters)); #conv filters 
	push!(weights, zeros(1,1)); # visible bias
	push!(weights, zeros(1,1,num_filters,1)); #hidden biases, will they be different for each channel if there are multiple channels? should probably be
	
	crbm = Dict{String, Any}();
	crbm["num_hidden"] = num_hidden
	crbm["num_filters"] = num_filters
	crbm["weights"] = weights
	crbm["learning_rate"] = learning_rate
	crbm["target_sparsity"] = target_sparsity
	#keep in mind that  sparsity lr is relative to the lerning rate since they are multiplied 
	crbm["sparsity_lr"] = sparsity_lr
	crbm["num_channels"] = num_input_channels
	
	return crbm
end

function train(crbm_, batches; max_epochs = 1000)
	crbm = deepcopy(crbm_)
	num_instances = size(batches[1], 4)
	num_batches = size(batches,1)
	
	for epoch in 1:max_epochs
		batch_index = epoch % num_batches + 1
		data = batches[batch_index]
		# expected data dims (input_width, input_height, num_channels, num_instances(batchsize))
		
		# note: we are using the activatiton probabilities instead of the states themselves
		
		pos_hidden_states, pos_hidden_probs = positive_phase_with_conv2(crbm, data)
		neg_visible_states, neg_visible_probs = negative_phase_with_conv2(crbm, pos_hidden_states)
		
		#second positive phase, resampling of hidden
		neg_hidden_states, neg_hidden_probs = positive_phase_with_conv2(crbm, neg_visible_probs)
		
		
		# pos_associations = appr. of expected value of derivative of E(data, hidden) given training data, over hidden values
		# neg_associations = appr. of expected value of derivative of E(data, hidden) given training data, over both hidden and visible values
		
		# if there are multiple channels, we will probably need to loop over channels as well
		for filter_index in 1:crbm["num_filters"]
			sum_diff_associations = 0
			sum_diff_probs = 0
			
			for instance_index in 1:num_instances
				# hidden values given training data (here using probs) * transpose of data
				p_h_p = rot180(pos_hidden_probs[:,:,filter_index, instance_index])
				d = data[:,:, 1, instance_index]
				offset_w = size(p_h_p, 1) - 1
				offset_h = size(p_h_p, 2) - 1
				pos_associations = conv2(p_h_p, d)[offset_w + 1 : end - offset_w, offset_h + 1 : end - offset_h]
				
				n_h_p = rot180(neg_hidden_probs[:,:,filter_index, instance_index])
				n_v_p = neg_visible_probs[:,:,1,instance_index]
				neg_associations = conv2(n_h_p, n_v_p)[offset_w + 1 : end - offset_w, offset_h + 1 : end - offset_h]
				
				# the update term per instance
				sum_diff_associations += pos_associations - neg_associations
				# sparsity target - 1/numhidden sq sum of probs for each filter
				# should we use neg or pos hidden probs
				sum_diff_probs += (sum(p_h_p - n_h_p) / (crbm["num_hidden"]^2)) + crbm["sparsity_lr"] * (crbm["target_sparsity"] - sum(p_h_p) / (crbm["num_hidden"]^2))
			end
			
			# update for the filter
			crbm["weights"][1][:,:,1,filter_index] += crbm["learning_rate"] * (sum_diff_associations / (crbm["num_hidden"]^2)) / num_instances
			crbm["weights"][3][1,1,filter_index, 1] += crbm["learning_rate"] * (sum_diff_probs ) / num_instances	
		end
		
		crbm["weights"][2][1,1] += crbm["learning_rate"] * ((sum(data - neg_visible_probs) / (size(data,1) * size(data, 2))) / num_instances)
		error = sum((data - neg_visible_probs).^2)
		print("Epoch ", epoch, ": error is ", (batch_index, error), "\n")
	end
	
	return crbm
end

function positive_phase_with_conv4(crbm, data) 
		# positive phase, sampling hidden layer from visible. 
		# for kth filter, h_ij^k = sigm((transpose(W^k) * v)_ij + b_k) -- post_hidden_probs
		
		
		# the convolution will convolve for all k filters
		# result will have size (1 + input_width - filter_width,  1 + input_height - filter_height, num_filters, num_instances)
		# valid cross correlation? (since the filters should be flipped)
		pos_hidden_activations = conv4(rbm["weights"][1], data; mode=1) 
		pos_hidden_probs = sigm(pos_hidden_activations .+ crbm["weights"][3])
		
		hidden_width = 1 + size(data, 1) - crbm["num_hidden"]
		hidden_height  = 1 + size(data, 2) - crbm["num_hidden"]
		pos_hidden_states = pos_hidden_probs .> rand(hidden_width, hidden_height, crbm["num_filters"], num_instances)
		pos_hidden_states = convert(Array{Float64}, pos_hidden_states)

		return pos_hidden_states, pos_hidden_probs
end

# assuming single channel
function positive_phase_with_conv2(crbm, data)
	input_width = size(data,1)
	input_height = size(data,2)
	num_instances = size(data, 4)
	
	pos_hidden_activations = zeros(1 + input_width - crbm["num_hidden"],  1 + input_height - crbm["num_hidden"], crbm["num_filters"], num_instances)
	
	for filter_index in 1:crbm["num_filters"]
		filter = rot180(reshape(crbm["weights"][1][:,:,1,filter_index], crbm["num_hidden"], crbm["num_hidden"]))
		
		for instance_index in 1:size(data, 4)
			instance = reshape(data[:,:, 1, instance_index], size(data, 1), size(data, 2))

			# for conv2 default is full convolution instead of valid, so the result we get is 2 *crbm["num_hidden"] - 1 bigger
			# we need to crop from both sides
			offset = crbm["num_hidden"] - 1
			convolution = conv2(filter, instance)
			pos_hidden_activations[:,:, filter_index, instance_index] = convolution[offset + 1: end - (offset), offset + 1 : end - offset]
		end
	end
	
	#below is same as above function, remove duplicate
	pos_hidden_probs = sigm(pos_hidden_activations .+ crbm["weights"][3])
		
	hidden_width = 1 + size(data, 1) - crbm["num_hidden"]
	hidden_height  = 1 + size(data, 2) - crbm["num_hidden"]
	pos_hidden_states = pos_hidden_probs .> rand(hidden_width, hidden_height, crbm["num_filters"], num_instances)
	pos_hidden_states = convert(Array{Float64}, pos_hidden_states)

	return pos_hidden_states, pos_hidden_probs
end

function negative_phase_with_conv4(crbm, pos_hidden_states)
		# negative phase reconstruction od visible units
		# i want the result to have the same size as the sample dims (input_width, input_height, num_channels, num_instances)
		# the result of regular convolution: (1 + (1+ input_width - filter_width) - filter_width, 1 + (1 + input_height - filter_height) - filter_height, num_filters, num_instances)
		# but since i actually want a full conv. ie (-1 + (1 + input_width - filter_width) + filter_width), so I need padding
		# the difference between 2 is 2 * (filter_width - 1)
		# then i will sum over the filters dimension, the third one
		
		full_padding = crbm["num_hidden"] - 1
		neg_visible_activations = sum(conv4(crbm["weights"][1], pos_hidden_states; padding = full_padding),3)
		neg_visible_probs = sigm(neg_visible_activations .+ crbm["weights"][2])
		visible_states = neg_visible_probs .> rand(visible_width, visible_height, crbm["num_channels"], num_instances)
		visible_states = convert(Array{Float64}, visible_states)

		return visible_states, neg_visible_probs
end

function negative_phase_with_conv2(crbm, pos_hidden_states)
	num_instances = size(pos_hidden_states, 4)
	
	hidden_width = size(pos_hidden_states, 1)
	hidden_height = size(pos_hidden_states, 2)
	visible_width = crbm["num_hidden"] +  hidden_width - 1
	visible_height = crbm["num_hidden"] + hidden_height -1

	neg_visible_activations = zeros(visible_width, visible_height, crbm["num_channels"], num_instances) 
	
	for instance_index in 1:num_instances
		filter_sum = 0
		for filter_index in 1:crbm["num_filters"]
			filter = reshape(crbm["weights"][1][:,:,1,filter_index], crbm["num_hidden"], crbm["num_hidden"])
			hidden = reshape(pos_hidden_states[:,:, filter_index, instance_index], hidden_width, hidden_height)

			filter_sum += conv2(filter, hidden)
		end

		neg_visible_activations[:,:,1,instance_index] = filter_sum
	end

	# below is same as conv4 version
	neg_visible_probs = sigm(neg_visible_activations .+ crbm["weights"][2])
	visible_states = neg_visible_probs .> rand(visible_width, visible_height, crbm["num_channels"], num_instances)
	visible_states = convert(Array{Float64}, visible_states)

	return visible_states, neg_visible_probs
end

function run_visible_hidden(rbm, data, is_visible)
	# if is visible, returns sample of hidden units from the visible units of a trained rbm
	# if not, returns sample of visible units from the hidden units of a trained rbm
	
	num_examples = size(data, 1)
	
	if is_visible
		num_units = rbm["num_hidden"]
		biases = rbm["weights"][3] # hidden biases
		weights = rbm["weights"][1]
	else
		num_units = rbm["num_visible"]
		biases = rbm["weights"][2] # visible biasaes
		weights = transpose(rbm["weights"][1])
	end
	
	states = ones(num_examples, num_units)
	
	activations = data * weights
	probs = sigm(activations .+ biases)
	states[:,:] = probs .> rand(num_examples, num_units)

	return states
end

function daydream(crbm, initial_visible, num_samples)
	# running gibbs sampling initialized from a given sample
	# takes samples in each reconstruction of the visible units
	
	samples  = ones(size(initial_visible,1), size(initial_visible, 2), size(initial_visible, 3), num_samples)
	samples[:,:, :, 1] = initial_visible
	
	# keeps hidden units as binary but visible units as real (probabilities)
	for row in 2 : num_samples
		visible = reshape(samples[:,:, 1, row - 1], size(initial_visible, 1), size(initial_visible, 2), size(initial_visible, 3), 1)
		
		hidden_states, hidden_probs = positive_phase_with_conv2(crbm, visible)
		visible_states, visible_probs = negative_phase_with_conv2(crbm, hidden_states)
		
		samples[:,:,:,row] = visible_states
	end
	
	return samples
end

end