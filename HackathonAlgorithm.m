
data = readtable("train.csv");

data1 = readtable("dataTest.txt");

load('HackathonModel.mat')

%  Input: Trained Model 1
[trainedClassifier, validationAccuracy] = trainClassifier(data1)
donate = trainedModel.predictFcn(data1);
donate
csvwrite('DidTheyDonate.txt',donate)



