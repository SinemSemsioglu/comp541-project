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
    models, states, recons  = TRAIN.main(images=xtrn[:,:,:,1:1000], numlayers=2, filtersize=[12,6], numfilters=[40,40],save_model_data=1, model_path="mnist_1000_adam_lr_0.1_slr_50_max_epoch_10000_10000.jld", gradient_lr = 0.1, sparsity_lr = 50, max_iterations=[10000,10000], debug=1)

    ACTIVATIONS.write_training_data("mnist_1000_adam_0.06_100_10000_10000.libsvm", recons, ytrn[:,1:1000],2,1 )

    train_svm("mnist_1000_adam_0.06_100_10000_10000.libsvm", "mnist_1000_adam_0.06_100_10000_10000.model", 1, ytrn[:,1:1000])

    #get activations and the libsvm file written
#ACTIVATIONS.main(;images=xtrn[:,:,:,:], model=models, numlayers=2, pool=[2,2],save_feature_info=1,feature_path="mnist_all.libsvm",labels=ytrn[:,:],one_hot=1)

end

function train_svm(libsvm_file_path, svm_model_path, debug, labels)
    #SVM training
    x,y = SVR.readlibsvmfile(libsvm_file_path)
    pmodel = SVR.train(y, x');

    if debug == 1
        y_pr = SVR.predict(pmodel, x')

        y_gold = WRITE_LIBSVM.convert_one_hot_to_class(labels)
        rounded = map(x-> Int(round(x)), y_pr)
        num_accurate = sum(map(x -> x ? 1: 0, rounded .== y_gold))
        print("training accuracy: ", num_accurate/ size(labels,2))
    end

    SVR.savemodel(pmodel, svm_model_path)
    SVR.freemodel(pmodel)
end

function test_svm()

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
