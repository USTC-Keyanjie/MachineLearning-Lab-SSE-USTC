
function svm_iris()
  
% Load training features and labels
[ train_y , train_x ] = libsvmread ( 'iris_data_train.txt' );

gamma = 1;

% Libsvm options
% -s 0 : classification
% -t 2 : RBF kernel
% -g : gamma in the RBF kernel
model = svmtrain(train_y, train_x, sprintf('-s 0 -t 2 -g %g', gamma));

% Load testing features and labels
[ test_y , test_x ] = libsvmread ( 'iris_data_test.txt' );

% Display training accuracy
[predicted_label, accuracy, decision_values] = svmpredict(test_y, test_x, model);

% Plot training data and decision boundary
% plotboundary(y, x, model);
% title(sprintf('\\gamma = %g', gamma), 'FontSize', 14);