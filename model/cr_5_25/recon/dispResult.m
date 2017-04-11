clear

load 'reconResult.mat'

disp(num2str(size(modelOut)));
disp(num2str(size(labelOut)));

hei = 160; 
wid = 160;
dimImage = 32;
numChannels = 1;
seqLength = 10;
numImageBlk = size(modelOut, 5);
blkDim = hei/dimImage;
blkImage = blkDim^2;
numImage = floor(numImageBlk / blkImage);

seqFrame = zeros(numImage, hei, wid, numChannels, seqLength);
frame = zeros(hei, wid, seqLength, numImage);
for i = 1:numImage
	seqImage = modelOut(:,:,:,:,(i-1)*blkImage+1:i*blkImage);
	seqLabel = labelOut(:,:,:,:,(i-1)*blkImage+1:i*blkImage);
	for j = 1:seqLength
		image = seqImage(:,:,:,j,:);
		image = squeeze(image);
		label = seqLabel(:,:,:,j,:);
		label = squeeze(label);
		
		for p = 1:blkDim
			for q = 1:blkDim
				frame((p-1)*dimImage+1:p*dimImage,(q-1)*dimImage+1:q*dimImage,j,i) ...
					= image(:,:,(p-1)*blkDim+q);
				frameLabel((p-1)*dimImage+1:p*dimImage,(q-1)*dimImage+1:q*dimImage,j,i) ...
					= label(:,:,(p-1)*blkDim+q);
			end
		end
	end
end

psnrAcc = 0;
ssimAcc = 0;
psnrDebug1 = 0;
psnrDebug2 = 0;
ssimDebug1 = 0;
ssimDebug2 = 0;

v = VideoWriter('video_1_25.avi');
v.FrameRate = 5;
open(v)

fig = figure(1);
for i = 1:size(frame, 4)
	for j = 1:size(frame,3)
		img = frame(:,:,j,i);
		img = im2uint8(img);
		
		lab = frameLabel(:,:,j,i);
		lab = im2uint8(lab);
		
		subplot(121)
		imshow(img);
		title(['current psnr: ' num2str(psnr(img, lab)), 'frame:', num2str(j)])
		
		subplot(122)
		imshow(lab);
		psnrAcc = psnrAcc + psnr(img, lab);
		ssimAcc = ssimAcc + ssim(img, lab);
		title(['current ssim:', num2str(ssim(img, lab))]);
		cFrame = getframe(fig);
		writeVideo(v, cFrame);

		if j==1
			psnrDebug1 = psnrDebug1 + psnr(img, lab);
			ssimDebug1 = ssimDebug1 + ssim(img, lab);
		else
			psnrDebug2 = psnrDebug2 + psnr(img, lab);
			ssimDebug2 = ssimDebug2 + ssim(img, lab);
		end
	end
end
close(v)

psnrAvg = psnrAcc / (size(frame,4)*size(frame,3));
psnrDebug1 = psnrDebug1 / (0.1*(size(frame,4)*size(frame,3)));
psnrDebug2 = psnrDebug2 / (0.9*(size(frame,4)*size(frame,3)));

ssimAvg = ssimAcc / (size(frame,4)*size(frame,3));
ssimDebug1 = ssimDebug1 / (0.1*(size(frame,4)*size(frame,3)));
ssimDebug2 = ssimDebug2 / (0.9*(size(frame,4)*size(frame,3)));





