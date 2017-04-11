%data1-feature1, data2 1-original label 1-10
function GenerateTestData(cr)
	task = 'Test';
	addpath('/home/user/kaixu/myGitHub/caffe/matlab/');

	if cr == 5
		load /home/user/kaixu/myGitHub/VideoReconNet/phi/phi3_cr5.mat
		phi1 = phi3;
	elseif cr == 25
		load /home/user/kaixu/myGitHub/VideoReconNet/phi/phi3_cr25.mat
		phi1 = phi3;
	elseif cr == 50
		load /home/user/kaixu/myGitHub/VideoReconNet/phi/phi3_cr50.mat
		phi1 = phi3;
	elseif cr == 100
		load /home/user/kaixu/myGitHub/VideoReconNet/phi/phi3_cr100.mat
		phi1 = phi3;
	end

	phase = 'test';

	%%
	if cr == 5
		net_model = 'videoNetCNN_cr5_deploy_10172016.prototxt';
		net_weights = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr5/'...
					'Snapshots/cr_5_CNN_10172016/videoNetCNN_5_iter_175000.caffemodel'];
	elseif cr == 25
		net_model = ['videoNetCNN_cr25_deploy_10172016.prototxt'];
		net_weights = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr25/'...
					'Snapshots/cr_25_CNN_10172016/videoNetCNN_25_iter_186549.caffemodel'];
	elseif cr == 50
		net_model = 'videoNetCNN_cr50_deploy_10172016.prototxt';
		net_weights = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr50/'...
					'Snapshots/cr_50_CNN_10172016/videoNetCNN_50_iter_338535.caffemodel'];
	elseif cr == 100
		net_model = 'videoNetCNN_cr100_deploy_10172016.prototxt';
		net_weights = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/models/cr100/'...
					'Snapshots/cr_100_CNN_10172016/videoNetCNN_100_iter_320361.caffemodel'];
	end

	%%
	folder = ['~/myGitHub/datasets/UCF101/', task, 'Data/5_196/test1/group1/'];
	savepath = ['data/',task,'Data_',num2str(cr),'_10172016.h5'];

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

	size_input = 32;
	size_label = 32;
	crop = 18;
	stride = size_input;
	seq_length = 10;	% length of the LSTM

	data = zeros(size(phi1,1), 1, numChannels, 1, 'single');
	label = zeros(size_label, size_label, numChannels, 1, 'single');

	% 	padding = abs(size_input - size_label)/2;
	count = 1;

	im_label = zeros(imgHeight-2*crop, imgWidth-2*crop, numChannels);
	subim_input = zeros(size_input, size_input, numChannels);
	subim_label = zeros(size_input, size_input, numChannels);
	subim_input_dr = zeros(size(phi1,1), 1, numChannels);
	modelOut = zeros(size_input, size_input, numChannels, 1);

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

		for j = 1:1:numImage
				rawImg = imread([folder, newPaths(i).name, '/', filename(j).name]);
				rawImg = rgb2ycbcr(rawImg);
				rawImg = im2double(rawImg(:,:,1));
				rawImg = rawImg(crop+1:end-crop,crop+1:end-crop);
				[hei,wid, ~, ~] = size(rawImg);

			for x = 1 : stride : hei-size_input+1
				for y = 1 : stride : wid-size_input+1
					subim_input = rawImg(x : x+size_input-1, y : y+size_input-1, :);
					subim_input_rs = reshape(subim_input, [],numChannels);
					for xx = 1:numChannels
						subim_input_dr = phi1(:,:,xx) * subim_input_rs(:,xx);
					end
					subim_label = rawImg(x : x+size_label-1, y : y+size_label-1, :);

					data(:, :, :, count) = subim_input_dr;
					label(:, :, :, count) = subim_label;

					count=count+1;
				end
			end
		end
		disp([num2str(i/length(newPaths)*100),'%']);
	end
	
	id = 0;
	caffe.set_mode_gpu();
	caffe.set_device(id);
	net = caffe.Net(net_model, net_weights, phase);

	for i = 1:size(data, 4)
		feature = net.forward({data(:,:,:,i)});
		feature = feature{1};
		modelOut(:,:,:,i) = feature;
	end
	 
	numImages = size(modelOut,4)/25;
	numOrigImages = floor(size(modelOut, 4)/25)*25;
	frame = zeros(160, 160, 1, numImages);
	frameLabel = zeros(160, 160, 1, numImages);
	
	dimImage = 32;
	for i = 1:numImages
		for p = 1:5
			for q = 1:5
				frame((p-1)*dimImage+1:p*dimImage,(q-1)*dimImage+1:q*dimImage,1,i) ...
							= modelOut(:,:,:,(i-1)*25+(p-1)*5+q);
				frameLabel((p-1)*dimImage+1:p*dimImage,(q-1)*dimImage+1:q*dimImage,1,i) ...
							= label(:,:,:,(i-1)*25+(p-1)*5+q);
				end
		end
	end
	
	imwrite(frame(:,:,1,190), ['shotPut_recon_', num2str(cr), '.pgm']);
	imwrite(frameLabel(:,:,1,190), ['shotPut_base', num2str(cr), '.pgm']);

	imwrite(frame(:,:,1,210), ['skateboard_recon_', num2str(cr), '.pgm']);
	imwrite(frameLabel(:,:,1,210), ['skateboard_base_', num2str(cr), '.pgm']);
	
	imwrite(frame(:,:,1,70), ['guitar_recon_', num2str(cr), '.pgm']);
	imwrite(frameLabel(:,:,1,70), ['guitar_base_', num2str(cr), '.pgm']);
	
	psnrAcc = 0;
	ssimAcc = 0;
	elAcc = 0;
	
	index = 1;
	videoLength = [30,30,30,30,30,30,20,30];
	psnrAvg = zeros(length(videoLength), 1);
	ssimAvg = zeros(length(videoLength), 1);
	elAvg = zeros(length(videoLength), 1);

	for i = 1:length(videoLength)
		partFrame = frame(:,:,:,index:index+videoLength(i)-1);
		partLabel = frameLabel(:,:,:,index:index+videoLength(i)-1);
		index = index + videoLength(i);
		for j = 1:videoLength(i)
			img = partFrame(:,:,:,j);
			img = im2uint8(img);
			lab = partLabel(:,:,:,j);
			lab = im2uint8(lab);
			psnrAcc = psnrAcc + psnr(img, lab);
			ssimAcc = ssimAcc + ssim(img, lab);
			elAcc = elAcc + sum(abs(img(:)-lab(:)));
		end
		psnrAvg(i) = psnrAcc / videoLength(i);
		ssimAvg(i) = ssimAcc / videoLength(i);
		elAvg(i) = elAcc / (videoLength(i)*size(frame,3)*size(frame,2)*size(frame,1));

		psnrAcc = 0;
		ssimAcc = 0;
		elAcc = 0;
	end
save(['measurement', num2str(cr), '.mat'], 'psnrAvg', 'ssimAvg', 'elAvg');

end
