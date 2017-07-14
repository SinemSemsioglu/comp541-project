module RBM
using Knet

export init

function init(num_visible, num_hidden; learning_rate = 0.1, winit = 0.1)
	weights = Any[]
	push!(weights, winit * randn(num_visible, num_hidden));
	push!(weights, zeros(1,num_visible)); # visible biases
	push!(weights, zeros(1,num_hidden)); #hidden biases
	
	rbm = Dict{String, Any}();
	rbm["num_hidden"] = num_hidden
	rbm["num_visible"] = num_visible
	rbm["weights"] = weights
	rbm["learning_rate"] = learning_rate
	
	return rbm
end

function train(rbm_, batches; max_epochs = 1000)
	rbm = deepcopy(rbm_)
	num_examples = size(batches[1],1)
	num_batches = size(batches,1)
	
	for epoch in 1:max_epochs
		batch_index = epoch % num_batches + 1
		data = batches[batch_index]
		# note: we are using the activatiton probabilities instead of the states themselves
		
		# positive phase, sampling hidden layer from visible. 
		# h_j = sigm(W_j. x + b_j) -- post_hidden_probs
		# pos_associations = appr. of expected value of derivative of E(data, hidden) given training data, over hidden values
		# neg_associations = appr. of expected value of derivative of E(data, hidden) given training data, over both hidden and visible values
		
		pos_hidden_activations = data * rbm["weights"][1]
		pos_hidden_probs = sigm(pos_hidden_activations .+ rbm["weights"][3])
		pos_hidden_states = pos_hidden_probs .> rand(num_examples, rbm["num_hidden"])
		
		# hidden values given training data (here using probs) * transpose of data
		pos_associations = transpose(data) * pos_hidden_probs 
		
		# negative phase reconstruction od visible units
		neg_visible_activations = pos_hidden_states * transpose(rbm["weights"][1])
		neg_visible_probs = sigm(neg_visible_activations .+ rbm["weights"][2])
		
		#second positive phase, resampling of hidden
		neg_hidden_activations = neg_visible_probs * rbm["weights"][1]
		neg_hidden_probs = sigm(neg_hidden_activations .+ rbm["weights"][3])
		
		# hidden values given estimated value (here using probs) * transpose of estimated values
		neg_associations = transpose(neg_visible_probs) * neg_hidden_probs
		
		# update weights
		rbm["weights"][1] += rbm["learning_rate"] * ((pos_associations - neg_associations) / num_examples)
		rbm["weights"][3] += rbm["learning_rate"] * (sum(pos_hidden_probs - neg_hidden_probs,1) / num_examples)
		rbm["weights"][2] += rbm["learning_rate"] * (sum(data - neg_visible_probs, 1) / num_examples)
		
		error = sum((data - neg_visible_probs).^2)
		print("Epoch ", epoch, ": error is ", (batch_index, error), "\n")
	end
	
	return rbm
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

function daydream(rbm, initial_visible, num_samples)
	# running gibbs sampling initialized from a given sample
	# takes samples in each reconstruction of the visible units
	
	samples  = ones(num_samples, rbm["num_visible"])
	samples[1,:] = initial_visible
	
	# keeps hidden units as binary but visible units as real (probabilities)
	for row in 2 : num_samples
		visible = samples[row-1,:]
		
		hidden_activations = transpose(visible) * rbm["weights"][1]
		hidden_probs = sigm(hidden_activations .+ rbm["weights"][3])
		hidden_states = hidden_probs .> rand(1,rbm["num_hidden"])
		
		visible_activations = hidden_states * transpose(rbm["weights"][1])
		visible_probs = sigm(visible_activations .+ rbm["weights"][2])
		visible_states = visible_probs .> rand(1,rbm["num_visible"])
		samples[row, :] = visible_states
	end
	
	return samples
end

end
