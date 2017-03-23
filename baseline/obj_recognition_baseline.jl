using Knet,GZip,Compat

function main()
	datapath = "./Data/Caltech_101/"; #take as an argument
	batchsize = 100; # take as an argument
	
	data, labels, categoryDict = readData(datapath);
	
	xtrn, ytrn, xtst, ytst, dtrn, dtst = prepareDatasets(data, labels, batchsize);
	
	println("train accuracy: ", accuracy(dtrn));
	println("test accuracy: ", accuracy(dtst));
end

function prepareDatasets(data, labels, batchsize)
	numInstances = size(data,2);
	shuffled = randperm(numInstances);
	
	# training data set as 90% of the whole data, might need changing, might be an argument
	# discuss if it makes sense to have different sets of training/test data in each run
	# add separation for dev set
	numTrainInstances = round(Int, numInstances * 9 / 10);
	xtrn = data[:, shuffled[1:numTrainInstances]];
	ytrn = labels[:, shuffled[1:numTrainInstances]];
	xtst = data[:, shuffled[numTrainInstances + 1 : end]];
	ytst = labels[:, shuffled[numTrainInstances + 1 : end]]; 
	
	dtrn = minibatch(xtrn, ytrn, batchsize); 
    dtst = minibatch(xtst, ytst, batchsize);
	
	return xtrn, ytrn, xtst, ytst, dtrn, dtst
end

function readData(datapath) 
	#get category = directory names
	categories, categoryDict = getCategories(datapath);
	numClasses = size(categories,1);
	
	data = Any[];
	labels = Any[];
	
	#read images for each category
	for category in categories
		categoryId = categoryDict[category]; 
		images = getImages(datapath * category);
		if size(data,1) == 0
			data = images
		else
			data = [data images];
		end
		imageLabels = zeros(numClasses, size(images,2));
		imageLabels[categoryId, :] = 1;
		
		if size(labels,1) == 0
			labels = imageLabels
		else
			labels = [labels imageLabels];
		end
	end
	
	#might be useful to return categories as well?
	return data, labels, categoryDict
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

function predict(x)
	numClasses = 102;
	
	predictions = zeros(numClasses, size(x,2));
	for instanceIndex = 1: size(x,2)
		predictions[round(Int, rand() * (numClasses - 1)) + 1, instanceIndex] = 1;
	end
	
	return predictions;
end

function minibatch(x, y, batchsize) 
	data = Any[];
	numBatches = round(Int, size(x, 2) / batchsize);
	
	for batchIndex in 0:numBatches - 1
		xBatch = x[:, batchIndex * batchsize + 1: (batchIndex + 1) * batchsize];
		yBatch = y[:, batchIndex * batchsize + 1: (batchIndex + 1) * batchsize];
		push!(data,(xBatch, yBatch));
	end
	
	return data;
end

function getImages(path)
	imFiles = filter(x -> ismatch(r"\.jpg.gz", x), readdir(path));
	images = Any[];
	
	#need to normalize image sizes? they are all different
	for imFile in imFiles 
		# taken from lab
		f = gzopen(path * "/" * imFile)
		a = @compat read(f)
		close(f)
		a = convert(Array{Float32,1},a ./ 255);
		push!(images,a);
	end
	
	return images';
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

main()