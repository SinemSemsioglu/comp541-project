import TRAIN
using Compat, GZip

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
    TRAIN.main(;images=xtrn[:,:,:,1:10], numlayers=2, filtersize=[12,6], numfilters=[40,40],save_model_data=1, model_path="mnist_10.jld",save_feature_info=1,feature_path="mnist_10.libsvm",labels=ytrn[:,1:10],one_hot=1)
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
