module CRBM
using Knet, MAT
include("visualize_crbm.jl")
export init

#assume square filters
function init(opts)
	weights = Dict()
	num_input_channels = 1 # for now
	weights["w"] = opts[:winit] * randn(opts[:hidden][1], opts[:hidden][1], num_input_channels, opts[:numfilters]); #conv filters 
	weights["c"] = zeros(1,1); # visible bias
	weights["b"] = zeros(1,1,opts[:numfilters],1); #hidden biases, will they be different for each channel if there are multiple channels? should probably be
	
	crbm = Dict{String, Any}();
	crbm["num_hidden"] = opts[:hidden][1]
	crbm["num_filters"] = opts[:numfilters]
	crbm["weights"] = weights
	crbm["learning_rate"] = opts[:lr]
	crbm["target_sparsity"] = opts[:sparsityt]
	#keep in mind that  sparsity lr is relative to the lerning rate since they are multiplied 
	crbm["sparsity"] = opts[:sparsity]
	crbm["sparsity_lr"] = opts[:sparsitylr]
	crbm["num_channels"] = num_input_channels
	crbm["max_pooling"] = opts[:maxpool]
	crbm["pool_size"] = opts[:poolsize]
	crbm["opt"] = opts[:opt]
	
	if(crbm["opt"])
		# use adam bu default
		if(opts[:momentum])
			crbm["opt_method"] = "momentum"
		else
			crbm["opt_method"] = "adam"
		end
	end
	crbm["sigma"] = opts[:sigma]
	return crbm
end

function train(crbm_, batches; max_epochs = 1000)
	crbm = deepcopy(crbm_)
	num_instances = size(batches[1], 4)
	num_batches = size(batches,1)
	
	if crbm["opt"]
		#print("initialized adam params\n")

		opt_params = Dict()
		grads = Dict()

		for key in keys(crbm["weights"])
			if(crbm["opt_method"] == "adam")
				opt_params[key] = Adam(;lr=crbm["learning_rate"])
			elseif(crbm["opt_method"] == "momentum")
				opt_params[key] = Momentum(;lr=crbm["learning_rate"], gamma = 0.5)
			end

			grads[key] = zeros(size(crbm["weights"][key]))
		end
	end

	hidden_width = 1 + size(batches[1], 1) - crbm["num_hidden"]
	hidden_height  = 1 + size(batches[1], 2) - crbm["num_hidden"]
	crbm["hidden"] = zeros(hidden_width, hidden_height, crbm["num_filters"], num_instances)
	
	update_interval = Int(floor(max_epochs/20))

	if update_interval > 20
		update_interval = 20
	end

	for epoch in 1:max_epochs
		batch_index = epoch % num_batches + 1
		data = batches[batch_index]
		# expected data dims (input_width, input_height, num_channels, num_instances(batchsize))
		
		# note: we are using the activatiton probabilities instead of the states themselves
		
		pos_hidden_states, pos_hidden_probs, pos_pool_states = positive_phase_with_conv2(crbm, data)
		neg_visible_states, neg_visible_probs = negative_phase_with_conv2(crbm, pos_hidden_states)
		
		#second positive phase, resampling of hidden
		neg_hidden_states, neg_hidden_probs, neg_pool_states = positive_phase_with_conv2(crbm, neg_visible_probs)
		crbm["hidden"] = neg_hidden_states
		
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
				sum_diff_probs += (sum(p_h_p - n_h_p) / (crbm["num_hidden"]^2))
				
				if crbm["sparsity"]
					#print("adding sparsity terms\n")
					# sparsity target - 1/numhidden sq sum of probs for each filter
					# should we use neg or pos hidden probs
					sum_diff_probs += crbm["sparsity_lr"] * (crbm["target_sparsity"] - sum(p_h_p) / (crbm["num_hidden"]^2))
				end	
				
			end
			
			if crbm["opt"]
				#print("caculating gradz \n")
				w_grad = (sum_diff_associations / (crbm["num_hidden"]^2)) / num_instances
				b_grad = (sum_diff_probs ) / num_instances

				#if !crbm["max_pooling"]
				#	w_grad = -1 * w_grad
				#	b_grad = -1 * b_grad
				#end

				grads["w"][:,:,1,filter_index] = w_grad 
				grads["b"][1,1,filter_index,1] = b_grad
			else
				#print("doing things the old way\n")
				# update for the filter
				w_grad = crbm["learning_rate"] * (sum_diff_associations / (crbm["num_hidden"]^2)) / num_instances
				b_grad = crbm["learning_rate"] * (sum_diff_probs ) / num_instances

				#if crbm["max_pooling"]
				#	w_grad = -1 * w_grad
				#	b_grad = -1 * b_grad
				#end

				crbm["weights"]["w"][:,:,1,filter_index] += w_grad
				crbm["weights"]["b"][1,1,filter_index, 1] += b_grad
			end	
		end
		
		if crbm["opt"]
			#print("updating weights the adam way \n")
			c_grad = (sum(data - neg_visible_probs) / (size(data,1) * size(data, 2))) / num_instances

			#if !crbm["max_pooling"]
			#	c_grad = -1 * c_grad
			#end
			grads["c"][1,1] = c_grad 
			update!(crbm["weights"], grads, opt_params)
		else
			c_grad = crbm["learning_rate"] * ((sum(data - neg_visible_probs) / (size(data,1) * size(data, 2))) / num_instances)
			
			#if !crbm["max_pooling"]
			#	c_grad = -1 * c_grad
			#end

			crbm["weights"]["c"][1,1] += c_grad 
		end

		error = sum((data - neg_visible_probs).^2)
		if (error == NaN) break; end
		
		print("Epoch ", epoch, ": error is ", (batch_index, error), "\n")

		
		if epoch % update_interval == 0
			filters = VISUALIZE_CRBM.visualize(crbm["weights"]["w"], Int(crbm["num_filters"]/2), 2, true; path="")
			filename = string("epoch_", epoch, ".mat")
			file = matopen(filename, "w")
			curr_sparsity = calculate_average_sparsity(crbm["hidden"])
			write(file, "weights", crbm["weights"])
			write(file, "error", error)
			write(file, "filters", filters)
			#write(file, "hidden", crbm["hidden"])
			write(file, "sparsity", curr_sparsity)
			close(file)
		end
	end
	
	return crbm
end

function calculate_average_sparsity(hidden)
	total_sparsity = 0
	num_units = size(hidden,1) * size(hidden,2)

	# this can probably be done more cleverly
	for instance in 1:size(hidden,4)
		for filter in 1:size(hidden,3)
			total_sparsity += sum(hidden[:,:,filter,instance]) / num_units
		end
	end

	return total_sparsity / (size(hidden,3) * size(hidden,4))
end

function positive_phase_with_conv4(crbm, data) 
		# positive phase, sampling hidden layer from visible. 
		# for kth filter, h_ij^k = sigm((transpose(W^k) * v)_ij + b_k) -- post_hidden_probs
		
		
		# the convolution will convolve for all k filters
		# result will have size (1 + input_width - filter_width,  1 + input_height - filter_height, num_filters, num_instances)
		# valid cross correlation? (since the filters should be flipped)
		pos_hidden_activations = conv4(crbm["weights"]["w"], data; mode=1) 
		pos_hidden_probs = sigm(pos_hidden_activations .+ crbm["weights"]["b"])
		
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
		filter = rot180(reshape(crbm["weights"]["w"][:,:,1,filter_index], crbm["num_hidden"], crbm["num_hidden"]))
		
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
	energy = pos_hidden_activations .+ crbm["weights"]["b"]
		
	if (crbm["max_pooling"])
		#print("doin max poolin\n")
		# max pooling crbm activations
		energy = (1 / (crbm["sigma"]^2)) .* energy
		pos_hidden_probs, pos_hidden_states, pos_pool_probs, pos_pool_states = max_pool(energy, crbm["pool_size"])
		pos_hidden_probs = sigm(pos_hidden_probs)
	else
		#print("no doin max poolin\n")
		# regular crbm activations
		pos_hidden_probs = sigm(energy)
		
		hidden_width = 1 + size(data, 1) - crbm["num_hidden"]
		hidden_height  = 1 + size(data, 2) - crbm["num_hidden"]

		pos_hidden_states = pos_hidden_probs .> rand(hidden_width, hidden_height, crbm["num_filters"], num_instances)
		pos_pool_states = []
	end

	pos_hidden_states = convert(Array{Float64}, pos_hidden_states)
	pos_pool_states = convert(Array{Float64}, pos_pool_states)

	return pos_hidden_states, pos_hidden_probs, pos_pool_states
end

function negative_phase_with_conv4(crbm, pos_hidden_states)
		# negative phase reconstruction od visible units
		# i want the result to have the same size as the sample dims (input_width, input_height, num_channels, num_instances)
		# the result of regular convolution: (1 + (1+ input_width - filter_width) - filter_width, 1 + (1 + input_height - filter_height) - filter_height, num_filters, num_instances)
		# but since i actually want a full conv. ie (-1 + (1 + input_width - filter_width) + filter_width), so I need padding
		# the difference between 2 is 2 * (filter_width - 1)
		# then i will sum over the filters dimension, the third one
		
		full_padding = crbm["num_hidden"] - 1
		neg_visible_activations = sum(conv4(crbm["weights"]["w"], pos_hidden_states; padding = full_padding),3)
		neg_visible_probs = sigm(neg_visible_activations .+ crbm["weights"]["c"])
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
			filter = reshape(crbm["weights"]["w"][:,:,1,filter_index], crbm["num_hidden"], crbm["num_hidden"])
			hidden = reshape(pos_hidden_states[:,:, filter_index, instance_index], hidden_width, hidden_height)

			filter_sum += conv2(filter, hidden)
		end

		neg_visible_activations[:,:,1,instance_index] = filter_sum
	end

	# below is same as conv4 version
	energy = neg_visible_activations .+ crbm["weights"]["c"]
	if (crbm["max_pooling"])
		#print("doin max poolin\n")
		# max pooling crbm activations
		energy = (1/ (crbm["sigma"]^2)) .* energy
		neg_visible_probs, visible_states, neg_pool_probs, neg_pool_states = max_pool(energy, crbm["pool_size"])
		neg_visible_probs = sigm(neg_visible_probs)
	else
		#print("no doin max poolin\n")
		# regular crbm activations
		neg_visible_probs = sigm(energy)
		visible_states = neg_visible_probs .> rand(visible_width, visible_height, crbm["num_channels"], num_instances)
	end
	
	visible_states = convert(Array{Float64}, visible_states)

	return visible_states, neg_visible_probs
end

function run_visible_hidden(rbm, data, is_visible)
	# if is visible, returns sample of hidden units from the visible units of a trained rbm
	# if not, returns sample of visible units from the hidden units of a trained rbm
	
	num_examples = size(data, 1)
	
	if is_visible
		num_units = crbm["num_hidden"]
		biases = crbm["weights"]["b"] # hidden biases
		weights = crbm["weights"]["w"]
	else
		num_units = crbm["num_visible"]
		biases = crbm["weights"]["c"] # visible biasaes
		weights = transpose(crbm["weights"]["w"])
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
	initial_visible = rand(size(initial_visible))
	samples[:,:, :, 1] = initial_visible

	# keeps hidden units as binary but visible units as real (probabilities)
	for row in 2 : num_samples
		visible = reshape(samples[:,:, 1, row - 1], size(initial_visible, 1), size(initial_visible, 2), size(initial_visible, 3), 1)
		
		hidden_states, hidden_probs, pool_states = positive_phase_with_conv2(crbm, visible)
		visible_states, visible_probs = negative_phase_with_conv2(crbm, hidden_states)
		
		samples[:,:,:,row] = visible_states
	end
	
	return samples
end

function max_pool(energy, pool_size)
	# multivariate sampling of hidden elements + computing pool group

	# zero padding in order to avoid calc problems when size is not div. by pool size
	off_x = ((pool_size - (size(energy, 1) % pool_size)) % pool_size) / 2
	off_y = ((pool_size - (size(energy,2) % pool_size)) % pool_size) / 2
	energies = zeros(Int(off_x * 2) + size(energy,1), Int(off_y * 2) + size(energy,2), size(energy,3), size(energy,4))
	energies[1 + Int(floor(off_x)):end - Int(ceil(off_x)), 1 + Int(floor(off_y)):end - Int(ceil(off_y)), :, :] = energy

	hidden_s = zeros(size(energies))
	hidden_p = zeros(size(energies))
	num_pools_x = Int(size(energies,1) / pool_size) # number of horizontal divisions for each hidden unit group
	num_pools_y = Int(size(energies,2) / pool_size)
	
	num_pools = num_pools_x * num_pools_y * size(energies,3) * size(energies,4)
	# configs represent all possibilities for hidden unit values for each pooling unit
	configs = zeros(pool_size^2 + 1, num_pools)
	
	for col in 1:pool_size
		for row in 1:pool_size
			# finds all ((x_pool-1) * pool_size + y_pool)th elements in pool groups
			# and puts them in config (ie the prob of the first element being on)
			elms = energies[row:pool_size:end, col:pool_size:end, :, :]
			configs[(col-1) * pool_size + row, :] = reshape(elms, 1, num_pools)
		end
	end

	configs = configs .- maximum(configs,1);
	exp_conf = exp(configs)

	# convert exponentiated values into real probabilities
	# these correspond to probability of each pool unit being on and all being off
	probs = transpose(exp_conf ./ sum(exp_conf,1))
	# cumulative sum of probabilties compared to a random number for multivariate sampling
	compare_prob = map(x -> x? 1:0, cumsum(probs, 2) .> rand(size(probs,1),1))
	# the on unit for each pool group the index of 1 reflects the real index -1
	sampled_indices_prev = diff(compare_prob,2)
	sampled_indices = zeros(size(probs))
	# here we turn the first unit on, if the first element in compare_prob was 1
	# then the there wont' be any difference with a value of 1
	sampled_indices[:,1] = 1 - sum(sampled_indices_prev,2)
	sampled_indices[:,2:end] = sampled_indices_prev
	samples = transpose(sampled_indices)
	probs = transpose(probs)
	
	# setting the hidden unit values
	for col in 1:pool_size
		for row in 1:pool_size
			selected_samples = reshape(samples[(col-1)*pool_size + row, :], num_pools_x, num_pools_y, size(energies, 3), size(energies,4))
			selected_probs = reshape(probs[(col-1)*pool_size + row, :], num_pools_x, num_pools_y, size(energies, 3), size(energies,4))
			hidden_s[row:pool_size:end, col:pool_size:end, :, :] = selected_samples
			hidden_p[row:pool_size:end, col:pool_size:end, :, :] = selected_probs
		end
	end

	# we need to trim the hidden states and probs before returning (in case zero padded)
	hidden_states = hidden_s[1 + Int(floor(off_x)):end - Int(ceil(off_x)), 1 + Int(floor(off_y)):end - Int(ceil(off_y)), :, :]
	hidden_probs = hidden_p[1 + Int(floor(off_x)):end - Int(ceil(off_x)), 1 + Int(floor(off_y)):end - Int(ceil(off_y)), :, :]

	# setting the pool unit values
	pool_states_row = sum(samples[1:end-1,:],1)
	pool_probs_row = sum(probs[1:end-1,:],1)

	pool_states = reshape(pool_states_row, num_pools_x, num_pools_y, size(energies,3), size(energies,4))
	pool_probs = reshape(pool_probs_row, num_pools_x, num_pools_y, size(energies,3), size(energies,4))

	return hidden_probs, hidden_states, pool_probs, pool_states
end

end

# want to sum probs for each pool group pool_size x pool_size
	# for the positive phase (sampling hidden) we want to do this for each filter separately
	# the hid_probs - hidden_width, hidden_height, crbm["num_filters"], num_instances
	# for the negative phase (sampling visible) we have just one layer
	# visible_width, visible_height, crbm["num_channels"], num_instances
	# in this phase we don't need to compute the pool layer
	#
	#for instance in 1: size(hid_probs,4)
	#	for layer in 1: size(hid_probs,3)
	#		for x_pool in 1:num_pools_x
	#			for y_pool in 1:num_pools_y
	#				x_indices = ((x_pool-1) * pool_size) + 1: (x_pool * pool_size)
	#				y_indices = ((y_pool-1) * pool_size) + 1 : (y_pool * pool_size)
	#				pool_sum = sum(exp_probs[x_indices, y_indices, layer, instance])
	#				pool[x_pool, y_pool, layer, instance] = 1/ (1 + pool_sum)
	#				hidden[x_indices, y_indices, layer, instance] = exp_probs[x_indices, y_indices, layer, instance] /(1+ pool_sum)				
	#			end
	#		end
	#	end
	#end

	
	#if sum(hidden) == NaN || sum(pool) == NaN
	#	file = matopen("nan.mat", "w")
	#	write(file, "hidden", hidden, "pool", pool)
	#	close(file)
	#end
