% DTMF 2 for LIBSVM 

% LIBSVM (Matlab version) DTMF
addpath('/Users/huckyang/Desktop/libsvm-3.22/matlab');

% training data and labels
features_a=[ 697.234, 770.143, 784.996, 684.997, 655.066, 625.093, 794.999, 785.037, 775.277, 689.192, 695.038, 695.125, 785.117, 783.943, 775.027, 684.996;
            1209.503,1336.439,1321.517,1211.545,1211.586,1221.533,1321.643,1321.451,1321.569,1201.545,1201.564,1201.493,1321.639,1321.540,1321.525,1207.463]';
% first column: the first tone feature  
% second column: the second tone feature 
label_a = [1,5,5,1,1,1,5,5,5,1,1,1,5,5,5,1]';
% test data and labels
features_b=[ 695.046, 784.995, 795.176, 785.068, 695.111;
            1201.518,1321.485,1321.436,1321.654,1201.560]';
label_b = [1,5,5,5,1]';
% scaling
[m,N]=size(features_a);
[m1,N]=size(features_b);
mf=mean(features_a);
nrm=diag(1./std(features_a,1));
features_1=(features_a-ones(m,1)*mf)*nrm;
features_2=(features_b-ones(m1,1)*mf)*nrm;
% SVM
model = svmtrain(label_a, features_1,'-c 1 -g 0.07');
% test
[predicted, accuracy, d_values] = svmpredict(label_b, features_2, model);
% predicted: the SVM output of the test data

