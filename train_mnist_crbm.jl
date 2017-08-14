for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end


include("crbm.jl")
include("visualize_crbm.jl")

using Knet   
using ArgParse # To work with command line argumands
using Compat,GZip # Helpers to read the MNIST (Like lab-2)

function main()

	#=
    In the macro, options and positional arguments are specified within a begin...end block
    by one or more names in a line, optionally followed by a list of settings. 
    So, in  the below, there are five options: epoch,batchsize,hidden size of mlp, 
    learning rate, weight initialization constant
    =#
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=100; help="number of epoch ")
        ("--batchsize"; arg_type=Int; default=100; help="size of minibatches")
        ("--numfilters"; arg_type=Int; default=40;help="number of filters")
        ("--hidden"; nargs='*'; arg_type=Int; default=[12];help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.5; help="learning rate")
        ("--sparsitylr"; arg_type=Float64; default=4.0; help="learning rate for sparsity term relative to lr")
        ("--sparsity"; arg_type=Float64; default=0.003; help="target sparsity for weights")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
    end
	
	 #=
    the actual argument parsing is performed via the parse_args function the result 
    will be a Dict{String,Any} object.In our case, it will contain the keys "epochs", 
    "batchsize", "hidden" and "lr", "winit" so that e.g. o["lr"] or o[:lr] 
     will yield the value associated with the positional argument.
     For more information: http://argparsejl.readthedocs.io/en/latest/argparse.html
    =#
    o = parse_args(s; as_symbols=true)
	
	# load the mnist data
    xtrnraw, ytrnraw, xtstraw, ytstraw = loaddata()
    
    inputSize = 28

    atype = (gpu()>=0 ? KnetArray{Float64} : Array{Float64})

    xtrn = convert(atype, reshape(xtrnraw ./ 255, inputSize, inputSize, 1, div(length(xtrnraw), inputSize ^ 2)))
    ytrnraw[ytrnraw.==0]=10;
    ytrn = convert(atype, sparse(convert(Vector{Int},ytrnraw),1:length(ytrnraw),one(eltype(ytrnraw)),10,length(ytrnraw)))
    xtst = convert(atype, reshape(xtstraw ./ 255, inputSize, inputSize, 1, div(length(xtstraw), inputSize^2)))
    ytstraw[ytstraw.==0]=10;
    ytst = convert(atype, sparse(convert(Vector{Int},ytstraw),1:length(ytstraw),one(eltype(ytstraw)),10,length(ytstraw)))
	
	dtrn = minibatch(xtrn, o[:batchsize])
    dtst = minibatch(xtst, o[:batchsize])
    
    ##
	
    crbm = CRBM.init(o[:hidden][1], o[:numfilters][1], 1; winit = o[:winit], learning_rate = o[:lr], target_sparsity= o[:sparsity], sparsity_lr = o[:sparsitylr])
	trained_crbm = CRBM.train(crbm, dtrn; max_epochs = o[:epochs])
	
	test_sample = dtst[1][:,:,:,50]
	num_samples = 20
	generated_samples = CRBM.daydream(trained_crbm, test_sample, num_samples)
	
    # assuming -- hidden is set as an even number
    #VISUALIZE_CRBM.visualize(trained_crbm["weights"][1], 5, 4, true; path="filters.mat")
	VISUALIZE_CRBM.visualize(trained_crbm["weights"][1], Int(o[:hidden]/2), 2, true; path="pool_filters.mat")
	VISUALIZE_CRBM.visualize(generated_samples, Int(num_samples/2), 2, false; path="pool_generated.mat")	
end

function loaddata()
	info("Loading MNIST...")
	xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
	xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
	ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
	ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]
	return (xtrn, ytrn, xtst, ytst)
end

function gzload(file; path="$file", url="http://yann.lecun.com/exdb/mnist/$file")
	isfile(path) || download(url, path)
	f = gzopen(path)
	a = @compat read(f)
	close(f)
	return(a)
end

# batches the the data X and not the labels
function minibatch(X, bs)
    data = Any[]
 
	num_batches = round(Int, size(X, 4) / bs);
	
	for batch_index in 0:num_batches - 1
		batch = X[:,:,:, batch_index * bs + 1: (batch_index + 1) * bs];
		push!(data, batch)
	end
	
	return data
end
