clear all;
cr = 1;
if cr == 1
	phi3 = zeros(1024, 1);
else
	dataFile = ['./phi/phi3_cr', num2str(cr), '.mat'];
	phi3 = load(dataFile);
	phi3 = phi3.phi3;
end
folder = '~/myGitHub/datasets/UCF101/TrainData/0.25_196/';
savepath = ['/home/user/kaixu/myGitHub/caffe/examples/videoNet/training/data/TrainDataCNN_', num2str(cr), '_10172016.h5'];
size_input = 32;
size_label = 32;
crop = 18;
stride = size_input;

data = zeros(size(phi3,1), 1, 1, 1);
label = zeros(size_label, size_label, 1, 1);
count = 0;

filepaths = dir(fullfile(folder,'*.jpg'));

for i = floor(linspace(1, length(filepaths), length(filepaths)))
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
	rawImg = image(crop+1:end-crop,crop+1:end-crop);
	im_label = rawImg;
	[hei,wid, ~, ~] = size(im_label);

	for x = 1 : stride : hei-size_input+1
		for y = 1 : stride : wid-size_input+1
				subim_input = im_label(x : x+size_input-1, y : y+size_input-1);
				if cr == 1
					subim_input =  subim_input(:);
				else
					subim_input = phi3(:,:,1)*subim_input(:);
				end

				subim_label = im_label(x : x+size_label-1, y : y+size_label-1);

				count=count+1;
				data(:, :, 1, count) = subim_input;
				label(:, :, 1, count) = subim_label;
		end
	end
	disp([num2str(i/length(filepaths)*100),'%']);
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order);

%% writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz);
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
