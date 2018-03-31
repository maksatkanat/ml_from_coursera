function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1; 
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
disp('THIS IS DATASET 3 PARAMS');

C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; 
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
errors = zeros(length(C_vec),length(sigma_vec));
minVal = size(Xval,1);
indi = 0;
indj = 0;
for i = 1:length(C_vec),
    c = C_vec(i);
    for j = 1:length(sigma_vec),
        sig = sigma_vec(j);
        model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig)); 
        predictions = svmPredict(model, Xval);
        errors(i,j) = sum(predictions~=yval);
        if errors(i,j)<minVal,
            minVal = errors(i,j);
            indi=i;
            indj=j;
        end;
% =========================================================================
    end;
end;
%disp(errors); % if you want to see no. of errors in each step
%minVal = min(min(errors));
%[i j] = find(errors == minVal); < Can't take indices
C = C_vec(indi);
sigma=sigma_vec(indj);
%disp(C);
%disp(sigma);
end
