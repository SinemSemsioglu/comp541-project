using Knet,GZip,Compat

function main()
	datapath = "./Data/Caltech_101"; #take as an argument
	batchsize = 100; # take as an argument
	
	#get category = directory names
	categories, categoryDict = getCategories(datapath);
	
	data = Any[];
	labels = Any[];
	#read images for each category
	for category in categories
		categoryId = categoryDict[category]; # look in the dict
		images = getImages(datapath * "/" * category);
		data = [data; images];
		imageLabels = zeros(size(images,1)) .+ categoryId;
		labels = [labels; imageLabels];
	end
	
	numInstances = size(data,1);
	shuffled = randperm(numInstances);
	
	# training data set as 90% of the whole data, might need changing, might be an argument
	# discuss if it makes sense to have different sets of training/test data in each run
	# add separation for dev set
	numTrainInstances = round(Int, numInstances * 9 / 10);
	xtrn = data[shuffled[1:numTrainInstances]];
	ytrn = labels[shuffled[1:numTrainInstances]];
	xtst = data[shuffled[numTrainInstances + 1 : end]];
	ytst = labels[shuffled[numTrainInstances + 1 : end]]; 
	
	dtrn = minibatch(xtrn, ytrn, batchsize); 
    dtst = minibatch(xtst, ytst, batchsize);
	
	println("train accuracy: ", accuracy(dtrn));
	println("test accuracy: ", accuracy(dtst));
end

function accuracy(dtst)
    ncorrect = 0
    ninstance = 0
	
    for (x,y) in dtst
		ninstance += size(x,1);
        ypred = predict(x);
		ncorrect += sum(y .== ypred);
    end
	
    return ncorrect/ninstance
end

function predict(x)
	numClasses = 101;
	predictions = Any[];
	for instanceIndex = 1: size(x,1)
		push!(predictions, round(Int, rand() * numClasses));
	end
	
	return predictions;
end

function minibatch(x, y, batchsize) 
	data = Any[];
	numBatches = round(Int, size(x, 1) / batchsize);
	
	for batchIndex in 0:numBatches - 1
		xBatch = x[batchIndex * batchsize + 1: (batchIndex + 1) * batchsize, :];
		yBatch = y[batchIndex * batchsize + 1: (batchIndex + 1) * batchsize, :];
		push!(data,(xBatch, yBatch));
	end
	
	return data;
end

function getImages(path)
	images = Any[];
	imFiles = filter(x -> ismatch(r"\.jpg.gz", x), readdir(path));
	for imFile in imFiles 
		# taken from lab
		f = gzopen(path * "/" * imFile)
		a = @compat read(f)
		close(f)
		a = convert(Array{Float32,1},a);
		push!(images,a);
	end
	
	return images;
end


function getCategories(path)
	#folder names correspond to categories, just be careful about Faces_Easy
	categories = readdir(path);
	
	#create a dict so that working with classes is easier?
	categoryDict = Dict{String, Int}();
	for catIndex = 1:size(categories,1)
		cat = categories[catIndex];
		categoryDict[cat] = catIndex;
	end
	
	return (categories, categoryDict);
end
