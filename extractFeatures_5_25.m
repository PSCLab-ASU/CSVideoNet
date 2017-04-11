function extractFeatures_5_25(task)

% extract raw CNN features per frame and save to feat_cache
%%
addpath('/home/user/kaixu/myGitHub/caffe/matlab/');

net_model_5 = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/' ...
			'videoNetCNN_cr5_model_deploy_feature.prototxt'];
net_weights_5 = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/'...
			  'Snapshots/cr_5_CNN_10072016/videoNetCNN_5_iter_390000.caffemodel'];
net_model_25 = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/' ...
			'videoNetCNN_cr25_model_deploy_feature.prototxt'];
net_weights_25 = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/'...
			  'Snapshots/cr_25_CNN_10072016/videoNetCNN_25_iter_145000.caffemodel'];
phase = 'test';

% task = 'Train';
% task = 'Val';
% testFile = '/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/ValData_5_25.h5';
% savepath = 'feature_conv5_5_25_test_80_test.h5';
if strcmp(task, 'Train') == 1
	testFile = './data/TrainData_5_25_10072016.h5';
	savepath = './data/Feature_conv5_5_25_train.h5';
elseif strcmp(task, 'Val') == 1
	testFile = './data/ValData_5_25_10072016.h5';
	savepath = './data/Feature_conv5_5_25_val.h5';
end


seqLength = 10;
numChannels = 16;
imHeight = 32;
imWidth = 32;
use_gpu = 1;

%%
if exist('use_gpu', 'var')
% 	try
% 		id = obtain_gpu_lock_id;
% 	catch e
% 		disp(e)
% 	end
	id = 1;
	caffe.set_mode_gpu();
	caffe.set_device(id);
	net_5 = caffe.Net(net_model_5, net_weights_5, phase);
	net_25 = caffe.Net(net_model_25, net_weights_25, phase);
	
% 	weight = net_5.blobs('conv5').get_data();
% 	weights_5 = net_5.copy_from(net_weights_5);
else
	caffe.set_mode_cpu();
end

%%
testData = h5read(testFile, '/data');
testLabel = h5read(testFile, '/label');

numData = size(testData, 5)/10;
if strcmp(task, 'Train') == 1
	numData = 256*80;	%300 - 40G	100 - 17G
end


chunksz = 128;
totalct = 0;
created_flag = false;

outputFeature = [];

for i = 1:numData
	data = testData(:,:,:,:,i);
	data_5 = data(:,:,:,1);
	data_25 = data(:,:,:,2:end);
	data_25 = data_25(1:floor(imHeight*imWidth/25),:,:,:);
	
	feature = net_5.forward({data_5});
	feature = feature{1};
	feature = permute(feature, [4 3 2 1]); 
	outputFeature_5 = feature;

	feature = net_25.forward({data_25});
	feature = feature{1};
	feature = permute(feature, [4 3 2 1]);
	outputFeature_25 = feature;

	outputFeature_5=reshape(outputFeature_5,[1,numChannels,imHeight,imWidth]);
	outputFeature_25=reshape(outputFeature_25,[9,numChannels,imHeight,imWidth]);
	tmp = [outputFeature_5;outputFeature_25];
	tmp = reshape(tmp, [1, size(tmp)]);
	outputFeature = [outputFeature;tmp];
	disp(['Completed',num2str(i/numData),'\n'])
	
	if mod(i, chunksz) == 0
		batchno = i / chunksz;
		last_read=(batchno-1) * chunksz;
		
		outputFeature = permute(outputFeature, [5 4 3 2 1]);
		batchdata = outputFeature;
		batchlabs = testLabel(:,:,:,:,last_read+1:last_read+chunksz);

		startloc = struct('dat',[1,1,1,1,totalct+1], 'lab', [1,1,1,1,totalct+1]);
		curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
		created_flag = true;
		totalct = curr_dat_sz(end);
		outputFeature = [];
	end
end


%% writing to HDF5


h5disp(savepath);

%%
caffe.reset_all();
% save('./data/outputFeature_5_25_conv5.mat', 'outputFeature');

function [feature] = caffe_forward(net, imseq)
	feature = net.forward({imseq});
	feature = feature{1};
	feature = permute(feature, [4 3 2 1]);
end

end
