clear
clc

cr = 100;
n = 1024;
phi=randn(floor(n/cr), n);
phi=orth(phi')';
phi3 = zeros(size(phi,1), size(phi,2), 3);
phi3(:,:,1) = phi;
phi3(:,:,2) = phi;
phi3(:,:,3) = phi;

fileName = ['./phi/phi3_cr', num2str(cr), '_', num2str(n)];
save(fileName, 'phi3');