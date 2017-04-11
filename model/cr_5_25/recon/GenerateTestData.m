%data1-feature1, data2 1-original label 1-10
function GenerateTestData(cr1, cr2)

task = 'Test';
addpath('/home/user/kaixu/myGitHub/caffe/matlab/');

folder = ['~/myGitHub/datasets/UCF101/',task, 'Data/5_196/test1/'];

paths = dir(folder);
paths(1:2) = [];
totalct = 0;
created_flag = false;

for i = 1:length(paths)
	totalct = GenerateGroupTrainData1ChanVarMix(cr1, cr2, num2str(i), task, totalct, created_flag);
	created_flag = true;
end

end
%%
function totalct = GenerateGroupTrainData1ChanVarMix(cr1, cr2, group, task, totalct_last, created_flag)
if cr1 == 5
	load ../../../phi/phi3_cr5.mat
	phi1 = phi3;
elseif cr1 == 25
	load ../../../phi/phi3_cr25.mat
	phi1 = phi3;
elseif cr1 == 50
	load ../../../phi/phi3_cr50.mat
	phi1 = phi3;
elseif cr1 == 100
	load ../../../phi/phi3_cr100.mat
	phi1 = phi3;
end

if cr2 == 5
	load ../../../phi/phi3_cr5.mat
	phi2 = phi3;
elseif cr2 == 25
	load ../../../phi/phi3_cr25.mat
	phi2 = phi3;
elseif cr2 == 50
	load ../../../phi/phi3_cr50.mat
	phi2 = phi3;
elseif cr2 == 100
	load ../../../phi/phi3_cr100.mat
	phi2 = phi3;
end

phase = 'test';

%%
if cr1 == 5
	net_model = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr5/'... ...
				'videoNetCNN_cr5_deploy_feature_10172016.prototxt'];
	net_weights = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr5/'...
			  'Snapshots/cr_5_CNN_10172016/videoNetCNN_5_iter_175000.caffemodel'];
elseif cr1 == 25
	net_model = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr25/'... ...
				'videoNetCNN_cr25_deploy_feature_10172016.prototxt'];
	net_weights = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr25/'...
			  'Snapshots/cr_25_CNN_10172016/videoNetCNN_25_iter_170000.caffemodel'];
end
id = 3;
caffe.set_mode_gpu();
caffe.set_device(id);
net = caffe.Net(net_model, net_weights, phase);

%%
folder = ['~/myGitHub/datasets/UCF101/', task, 'Data/5_196/test1/group', num2str(group), '/'];
savepath = ['data/',task,'Data_',num2str(cr1),'_',num2str(cr2),'_mix_feed_10172016.h5'];

if strcmp(task, 'Train') == 1
	step = 2;
elseif strcmp(task, 'Val') == 1
	step = 2;
else
	step = 1;
end

imgHeight = 196;
imgWidth = 196;
numChannels = 1;
cnnHidden = 16;

size_input = 32;
size_label = 32;
crop = 18;
stride = size_input;
seq_length = 10;	% length of the LSTM

% 	data = zeros(seq_length, numChannels, size(phi3,1), 1, 1, 'single');
% 	label = zeros(seq_length, numChannels, size_label, size_label, 1, 'uint8');
data1 = zeros(size_label, size_label, cnnHidden, 1, 1, 'single');
data2 = zeros(size(phi1,1), 1, numChannels, seq_length, 1, 'single');
label = zeros(size_label, size_label, numChannels, seq_length, 1, 'single');

% 	padding = abs(size_input - size_label)/2;
count = 1;

im_label = zeros(imgHeight-2*crop, imgWidth-2*crop, numChannels, seq_length);
subim_input = zeros(size_input, size_input, numChannels, seq_length);
subim_label = zeros(size_input, size_input, numChannels, seq_length);
subim_input_dr1 = zeros(size(phi1,1), 1, numChannels, 1);
subim_input_dr2 = zeros(size(phi1,1), 1, numChannels, seq_length-1);

paths = dir(folder);
paths(1:2) = [];

newPaths = [];

for i = 1:step:length(paths)
	newPaths = [newPaths,paths(i)];
end

% h = waitbar(0,'Writing train data to hdf5 file, please wait...');

for i = 1:length(newPaths)
	filename = dir([folder, newPaths(i).name]);
	filename(1:2) = [];	% filenames in each subfolder

	len = length(filename);
	tmp = floor(len / seq_length);
	if tmp >= 3
		tmp = 3;
	end
	numImage = tmp * seq_length;

	for j = 1:seq_length:numImage
		for k = 1:seq_length
			rawImg = imread([folder, newPaths(i).name, '/', filename(j+k-1).name]);
			rawImg = rgb2ycbcr(rawImg);
			rawImg = im2double(rawImg(:,:,1));
			rawImg = rawImg(crop+1:end-crop,crop+1:end-crop);
			im_label(:,:,:,k) = rawImg;
			[hei,wid, ~, ~] = size(im_label);
		end

		for x = 1 : stride : hei-size_input+1
			for y = 1 : stride : wid-size_input+1
				for z = 1 : seq_length
					subim_input(:, :, :, z) = im_label(x : x+size_input-1, y : y+size_input-1, :, z);
					subim_input_rs = reshape(subim_input, [],numChannels, seq_length);
					if z == 1
						for xx = 1:numChannels
							if size(phi1, 1) == size_input^2
								subim_input_dr1(:,:,xx,1) = subim_input_rs(:,xx,z);
							else
								subim_input_dr1 = phi1(:,:,xx) * subim_input_rs(:,xx,z);
								feature = net.forward({subim_input_dr1});
								feature = feature{1};
							end
						end
					else
						for xx = 1:numChannels
							tmp = phi2(:,:,xx) * subim_input_rs(:,xx,z);
							tmp(size(tmp,1)+1:size(phi1,1)) = 0;
							subim_input_dr2(:,:,xx,z-1) = tmp;
						end
					end

					subim_label(:, :, :, z) = im_label(x : x+size_label-1, ...
														y : y+size_label-1, :, z);
				end

				tmp = cat(4, subim_input_dr1, subim_input_dr2);
				data1(:, :, :, :, count) = feature;
				data2(:, :, :, :, count) = tmp;
				label(:, :, :, :, count) = subim_label;
				
				count=count+1;
			end
		end
	end
	disp(['group', num2str(group), ', ', num2str(i/length(newPaths)*100),'%']);

end

if strcmp(task, 'Train') ~= 0
	count = count - 1;
	order = randperm(count); 
	data1 = data1(:, :, :, :, order);
	data2 = data2(:, :, :, :, order);
	label = label(:, :, :, :, order);
end

% writing to HDF5
chunksz = 20;
% 	created_flag = false;
totalct = totalct_last;

for batchno = 1:floor(count/chunksz)
	last_read=(batchno-1) * chunksz;
	batchdata1 = data1(:,:,:,:,last_read+1:last_read+chunksz);
	batchdata2 = data2(:,:,:,:,last_read+1:last_read+chunksz);
	batchlabs = label(:,:,:,:,last_read+1:last_read+chunksz);

	startloc = struct('dat1',[1,1,1,1,totalct+1], 'dat2', [1,1,1,1,totalct+1], 'lab', [1,1,1,1,totalct+1]);
	curr_dat_sz = store2hdf5Mix(savepath, batchdata1, batchdata2, batchlabs, ~created_flag, startloc, chunksz); 
	created_flag = true;
	totalct = curr_dat_sz(end);
end
h5disp(savepath);

end


