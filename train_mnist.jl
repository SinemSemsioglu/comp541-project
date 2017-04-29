import TRAIN
using Compat, GZip, SVR

function main()
    path = "mnist/" # will be taken as an argument later since it depends on file organization
    inputSize = 28

    xtrnraw, ytrnraw, xtstraw, ytstraw = loaddata(path)

    xtrn = convert(Array{Float64}, reshape(xtrnraw ./ 255, inputSize, inputSize, 1, div(length(xtrnraw), inputSize ^ 2)))

    ytrnraw[ytrnraw.==0]=10;
    ytrn = convert(Array{Float64}, sparse(convert(Vector{Int},ytrnraw),1:length(ytrnraw),one(eltype(ytrnraw)),10,length(ytrnraw)))

    xtst = convert(Array{Float64}, reshape(xtstraw ./ 255, inputSize, inputSize, 1, div(length(xtstraw), inputSize^2)))

    ytstraw[ytstraw.==0]=10;
    ytst = convert(Array{Float64}, sparse(convert(Vector{Int},ytstraw),1:length(ytstraw),one(eltype(ytstraw)),10,length(ytstraw)))


    #train for each image
    models, states  = TRAIN.main(images=xtrn[:,:,:,:], numlayers=2, filtersize=[12,6], numfilters=[40,40],save_model_data=1, model_path="mnist_all_2.jld", gradient_lr = 0.018)

    #get activations and the libsvm file written
    ACTIVATIONS.main(;images=xtrn[:,:,:,:], model=models, numlayers=2, pool=[2,2],save_feature_info=1,feature_path="mnist_all.libsvm",labels=ytrn[:,:],one_hot=1)

    #SVM training
    x,y = SVR.readlibsvmfile("mnist_all.libsvm")
    pmodel = SVR.train(y, x');
    y_pr = SVR.predict(pmodel, x');

    y_gold = WRITE_LIBSVM.convert_one_hot_to_class(ytrn[:,:])
    rounded = map(x-> Int(round(x))), y_pr)
    num_accurate = sum(map(x -> x ? 1: 0, rounded .== y_gold))
    print("accuracy: ", num_accurate/size(xtrn,2))

    SVR.savemodel(pmodel, "mnist_svm.model")
    SVR.freemodel(pmodel)
end

function loaddata(datapath)
    info("Loading MNIST...")
    xtrn = gzload(datapath * "train-images.idx3-ubyte.gz")[17:end]
    xtst = gzload(datapath * "t10k-images.idx3-ubyte.gz")[17:end]
    ytrn = gzload(datapath * "train-labels.idx1-ubyte.gz")[9:end]
    ytst = gzload(datapath * "t10k-labels.idx1-ubyte.gz")[9:end]
    return (xtrn, ytrn, xtst, ytst)
end

function gzload(path)
    isfile(path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end
