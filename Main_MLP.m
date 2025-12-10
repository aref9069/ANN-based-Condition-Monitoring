clc;
clear all;
close all;

load KURTOSISfit;
load RMSfit;
load timeT5;
% load t5-chosenp;

% import inputs and targets
data=[KURfit;RMSfit];

inputs = data;
targets = time;

% Create a Fitting Network
hiddenLayerSize = 7;
TF={'tansig','purelin'};
net = newff(inputs,targets,hiddenLayerSize,TF);


net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};


% set dividing function
net.divideFcn = 'dividerand';  
net.divideMode = 'sample'; 
net.divideParam.trainRatio = 75/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% set train function
net.trainFcn = 'trainlm';  

% view(net)

% set performance function
net.performFcn = 'mse';  

% Choose Plot Functions
net.plotFcns = {'plotperform','ploterrhist','plotregression','plotfit'};

net.trainParam.showWindow=true;
net.trainParam.showCommandLine=false;
net.trainParam.show=1;
net.trainParam.epochs=100;
net.trainParam.goal=1e-8;
net.trainParam.max_fail=20;

% Train the Network
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);

% Recalculate Training, Validation and Test Performance
trainInd=tr.trainInd;
trainInputs = inputs(:,trainInd);
trainTargets = targets(:,trainInd);
trainOutputs = outputs(:,trainInd);
trainErrors = trainTargets-trainOutputs;
trainPerformance = perform(net,trainTargets,trainOutputs);

valInd=tr.valInd;
valInputs = inputs(:,valInd);
valTargets = targets(:,valInd);
valOutputs = outputs(:,valInd);
valErrors = valTargets-valOutputs;
valPerformance = perform(net,valTargets,valOutputs);

testInd=tr.testInd;
testInputs = inputs(:,testInd);
testTargets = targets(:,testInd);
testOutputs = outputs(:,testInd);
testError = testTargets-testOutputs;
testPerformance = perform(net,testTargets,testOutputs);



figure;
plotperform(tr);


% figure;
% plotfit(net,inputs,targets);

figure;
plotregression(trainTargets,trainOutputs,'Train Data',...
    testTargets,testOutputs,'Test Data',...
    valTargets,valOutputs,'Validation Data',...
    targets,outputs,'All Data')



