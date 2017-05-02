using Knet, JLD

import TRAIN, READ_IMAGES

function main()
    datapath = "101_ObjectCategories/"; #take as an argument
    batchsize = 100; # take as an argument

    data, labels, categoryDict = readData(datapath);

    xtrn, ytrn, xtst, ytst, dtrn, dtst = prepareDatasets(data, labels, batchsize);

    path = "natural_images/" # will be taken as an argument later since it depends on file organization
    images = readData(datapath)
    images .= mean(images, (1,2,3))

    models, states, recons = TRAIN.main(;images=images[:,:,:,:], debug=1, gradient_lr=0.02, sparsity_lr = 300, max_iterations=[124, 62], save_model_data=1, model_path="kyoto_adam_0.02_300_124_62_grayscale.jld")
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
        images = READ_IMAGES.get_square_color_images(datapath * category,1,150,1)
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
