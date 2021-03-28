function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
inputTable = trainingData;
predictorNames = {'Recency_months_', 'Frequency_times_', 'Monetary_c_c_Blood_', 'Time_months_'};
predictors = inputTable(:, predictorNames);
response = inputTable.ifTheyDonatedBloodOnTheDayOfTheSurvey;
isCategoricalPredictor = [false, false, false, false];
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Cosine', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));
trainedClassifier.RequiredVariables = {'Frequency_times_', 'Monetary_c_c_Blood_', 'Recency_months_', 'Time_months_'};
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2021a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');
inputTable = trainingData;
predictorNames = {'Recency_months_', 'Frequency_times_', 'Monetary_c_c_Blood_', 'Time_months_'};
predictors = inputTable(:, predictorNames);
response = inputTable.ifTheyDonatedBloodOnTheDayOfTheSurvey;
isCategoricalPredictor = [false, false, false, false];
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 5);
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
