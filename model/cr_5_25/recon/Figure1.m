function Figure1()
	load 'reconResult.mat'

	modelOut = reshape(modelOut, [sqrt(size(modelOut,1)), sqrt(size(modelOut,1)), size(modelOut,2), size(modelOut,3), size(modelOut,4) ]);
	labelOut = reshape(labelOut, [sqrt(size(labelOut,1)), sqrt(size(labelOut,1)), size(labelOut,2), size(labelOut,3), size(labelOut,4) ]);
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
	
	imwrite(frame(:,:,1,19), 'shotPut_recon_25_1.pgm');
	imwrite(frameLabel(:,:,1,19), 'shotPut_base_25_1.pgm');
	imwrite(frame(:,:,5,19), 'shotPut_recon_25_5.pgm');
	imwrite(frameLabel(:,:,5,19), 'shotPut_base_25_5.pgm');
	
	imwrite(frame(:,:,1,22), 'skateboard_recon_25_1.pgm');
	imwrite(frameLabel(:,:,1,22), 'skateboard_base_25_1.pgm');
	imwrite(frame(:,:,5,22), 'skateboard_recon_25_5.pgm');
	imwrite(frameLabel(:,:,5,22), 'skateboard_base_25_5.pgm');
	
	imwrite(frame(:,:,1,7), 'guitar_recon_25_1.pgm');
	imwrite(frameLabel(:,:,1,7), 'guitar_base_25_1.pgm');
	imwrite(frame(:,:,5,7), 'guitar_recon_25_5.pgm');
	imwrite(frameLabel(:,:,5,7), 'guitar_base_25_5.pgm');
	
	img = frame(:,:,1,19);
	img = im2uint8(img);
	lab = frameLabel(:,:,1,19);
	lab = im2uint8(lab);
	psnr1 = psnr(img, lab);
	ssim1 = ssim(img, lab);
	el1 = sum(abs(img(:)-lab(:))) / (size(img,1)*size(img,2));
	
	img = frame(:,:,5,19);
	img = im2uint8(img);
	lab = frameLabel(:,:,5,19);
	lab = im2uint8(lab);
	psnr5 = psnr(img, lab);
	ssim5 = ssim(img, lab);
	el5 = sum(abs(img(:)-lab(:))) / (size(img,1)*size(img,2));
	
	index = 1;
	videoLength = [3,3,3,3,3,3,2,2];
	psnrAvg = zeros(length(videoLength), 1);
	ssimAvg = zeros(length(videoLength), 1);
	elAvg = zeros(length(videoLength), 1);
	for i = 1:length(videoLength)
		frameSeq=frame(:,:,:,index:index+videoLength(i)-1);
		frameLabelSeq = frameLabel(:,:,:,index:index+videoLength(i)-1);
		[psnrAvg(i), ssimAvg(i), elAvg(i)] = measure(frameSeq, frameLabelSeq, 1);
		index = index + videoLength(i);
	end
	save('measurement.mat', 'psnrAvg', 'ssimAvg', 'elAvg');
end

function [psnrAvg, ssimAvg, elAvg] = measure(frame, frameLabel, disp)
	psnrAcc = 0;
	ssimAcc = 0;
	elAcc = 0;
	psnrDebug1 = 0;
	psnrDebug2 = 0;
	ssimDebug1 = 0;
	ssimDebug2 = 0;
	elDebug1 = 0;
	elDebug2 = 0;
	
	if disp == 1
		v = VideoWriter('video_1_25.avi');
		v.FrameRate = 5;
		open(v)
		fig = figure(1);
	end
	
	for i = 1:size(frame, 4)
		for j = 1:size(frame,3)
			img = frame(:,:,j,i);
			img = im2uint8(img);

			lab = frameLabel(:,:,j,i);
			lab = im2uint8(lab);
			
			psnrAcc = psnrAcc + psnr(img, lab);
			ssimAcc = ssimAcc + ssim(img, lab);
			elAcc = elAcc + sum(abs(img(:)-lab(:)));

			if disp == 1
				subplot(121)
				imshow(img);
				title(['current psnr: ' num2str(psnr(img, lab)), 'frame:', num2str(j)])
				subplot(122)
				imshow(lab);

				title(['current ssim:', num2str(ssim(img, lab))]);
				cFrame = getframe(fig);
				writeVideo(v, cFrame);
			end

			if j==1
				psnrDebug1 = psnrDebug1 + psnr(img, lab);
				ssimDebug1 = ssimDebug1 + ssim(img, lab);
			else
				psnrDebug2 = psnrDebug2 + psnr(img, lab);
				ssimDebug2 = ssimDebug2 + ssim(img, lab);
			end
		end
	end
	if disp == 1
		close(v)
	end

	psnrAvg = psnrAcc / (size(frame,4)*size(frame,3));
	psnrDebug1 = psnrDebug1 / (0.1*(size(frame,4)*size(frame,3)));
	psnrDebug2 = psnrDebug2 / (0.9*(size(frame,4)*size(frame,3)));

	ssimAvg = ssimAcc / (size(frame,4)*size(frame,3));
	ssimDebug1 = ssimDebug1 / (0.1*(size(frame,4)*size(frame,3)));
	ssimDebug2 = ssimDebug2 / (0.9*(size(frame,4)*size(frame,3)));
	
	elAvg = elAcc / (size(frame,4)*size(frame,3)*size(frame,2)*size(frame,1)) ;
	
end



