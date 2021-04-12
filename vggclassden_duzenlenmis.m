%% Transfer ��renmesi ile CNN a�lar�n�n e�itimi

% Ekran Temizleme

clc
clear all
close all


%Verilerin Okunmas�

    veri_yolu = fullfile('C:\Users\CASPER\Desktop\matconvnet-1.0-beta25\ar� veri seti');

    imds = imageDatastore(veri_yolu,...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    

% E�itim ve Test Verilerinin Ay�r�m�

[imdsTrain,imdsTest] = splitEachLabel(imds,0.75,'randomized');


% �ne�itmli A��n y�klenmesi 

net = vgg16;

% verilerin boyutland�r�lmas�

inputSize = net.Layers(1).InputSize


pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);


%�n e�itimli VGG-16 CNN a��n�n katmanlar�n�n g�ncellenmesi

layersTransfer = net.Layers(1:end-3);
numClasses=6;
layers = [
    layersTransfer
   fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)%set the learning rate of new layers
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];


%E�itim parametrelerinin ayarlanmas� 

maxEpochs = 30;
epochIntervals = 1;
initLearningRate = 0.1;
learningRateFactor = 0.1;
l2reg = 0.0001;
miniBatchSize = 2;
options = trainingOptions('sgdm', ...
    'InitialLearnRate',initLearningRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',10, ...
    'LearnRateDropFactor',learningRateFactor, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',augimdsTest, ...
    'ValidationFrequency',1000, ...
    'Plots','training-progress', ...
    'Verbose',false);

% E�itim i�lemi

[net tr] = trainNetwork(augimdsTrain,layers,options);
  
% Test ��lemi

[YValPred,probs] = classify(net,augimdsTest,'MiniBatchSize',10);

labelCount = countEachLabel(imds)

  actual((imdsTest.Labels)=='bee1')=1;
  actual(imdsTest.Labels=='bee2')=2;
  actual(imdsTest.Labels=='other_insect')=3;
  actual((imdsTest.Labels)=='other_noinsect')=4;
  actual(imdsTest.Labels=='waps1')=5;
  actual(imdsTest.Labels=='waps2')=6;
  
  
  predict(( YValPred)=='bee1')=1;
  predict( YValPred=='bee2')=2;
  predict( YValPred=='other_insect')=3;
  predict(( YValPred)=='other_noinsect')=4;
  predict( YValPred=='waps1')=5;
  predict( YValPred=='waps2')=6;
  



% E� olu�um matrislerinin eldesi

[c_matrix,Result,RefereceResult]= confusion.getMatrix(actual,predict);

%E� olu�um matrisi �izdirimi

figure;  
cm = confusionchart(imdsTest.Labels,YValPred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
cm.Title = 'DVM Algoritmas�nda Kullan�lan Test Verisi ��in E� Olu�um Matrisi';  

