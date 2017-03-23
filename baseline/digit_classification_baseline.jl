using Compat, GZip

function main()
	datapath = "./Data/MNIST/"; #take as an argument
	batchsize = 100; # take as an argument
	
	# Size of input vector (MNIST images are 28x28)
	inputSize = 28 * 28

	## Data loading & preprocessing
	#  Size of xtrn: (28,28,1,60000)
	#  Size of xtrn must be: (784, 60000)
	#  Size of xtst must be: (784, 10000)

	xtrnraw, ytrnraw, xtstraw, ytstraw = loaddata(datapath)
	
	xtrn = convert(Array{Float32}, reshape(xtrnraw ./ 255, inputSize, div(length(xtrnraw), 784)))
	ytrnraw[ytrnraw.==0]=10;
	ytrn = convert(Array{Float32}, sparse(convert(Vector{Int},ytrnraw),1:length(ytrnraw),one(eltype(ytrnraw)),10,length(ytrnraw)))
	
	xtst = convert(Array{Float32}, reshape(xtstraw ./ 255, inputSize, div(length(xtstraw), 784)))
	ytstraw[ytstraw.==0]=10;
	ytst = convert(Array{Float32}, sparse(convert(Vector{Int},ytstraw),1:length(ytstraw),one(eltype(ytstraw)),10,length(ytstraw)))

	dtrn = minibatch(xtrn, ytrn, batchsize); 
    dtst = minibatch(xtst, ytst, batchsize);
	
	println("train accuracy: ", accuracy(dtrn));
	println("test accuracy: ", accuracy(dtst));
end

function loaddata(datapath)
	info("Loading MNIST...")
	xtrn = gzload(datapath * "train-images-idx3-ubyte.gz")[17:end]
	xtst = gzload(datapath * "t10k-images-idx3-ubyte.gz")[17:end]
	ytrn = gzload(datapath * "train-labels-idx1-ubyte.gz")[9:end]
	ytst = gzload(datapath * "t10k-labels-idx1-ubyte.gz")[9:end]
	return (xtrn, ytrn, xtst, ytst)
end

function gzload(path)
	isfile(path)
	f = gzopen(path)
	a = @compat read(f)
	close(f)
	return(a)
end

function minibatch(x, y, batchsize) 
	data = Any[];
	numBatches = round(Int, size(x, 2) / batchsize);
	
	for batchIndex in 0:numBatches - 1
		xBatch = x[:,batchIndex * batchsize + 1: (batchIndex + 1) * batchsize];
		yBatch = y[:,batchIndex * batchsize + 1: (batchIndex + 1) * batchsize];
		push!(data,(xBatch, yBatch));
	end
	
	return data;
end

function predict(x)
	numClasses = 10 #either take as a parameter or make global variable if possible
	
	predictions = zeros(numClasses, size(x,2));
	for instanceIndex = 1: size(x,2)
		predictions[round(Int, rand() * (numClasses - 1)) + 1, instanceIndex] = 1;
	end
	
	return predictions;
end

function accuracy(dtst)
    ncorrect = 0
    ninstance = 0
	
    for (x,y) in dtst
		ninstance += size(x,2);
        ypred = predict(x);
		ncorrect += sum(y .* (ypred .== maximum(ypred, 1)));
	end
	
    return ncorrect/ninstance
end

main()

