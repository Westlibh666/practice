%% Training Data Generation 
%
% This script is created to generate training and validation data for the deep
% learning model in a single-user OFDM system. 

% The training data and the validation data is collected for a single
% subcarrier selected based on a pre-defined metric. The transmitter sends
% OFDM packets to the receiver, where each OFDM packet contains one OFDM
% pilot symbol and one OFDM data symbol. Data symbols can be interleaved in
% the pilot sequence. 

% Each training sample contains all symbols in a received OFDM packet and 
% is represented by a feature vector that follows the similar data struture
% in the MATLAB example of seqeunce classification using LSTM network.
% Please run the command and check it out.  
% >> openExample('nnet/ClassifySequenceDataUsingLSTMNetworksExample')

%% Clear workspace

clear variables;
close all;

%% OFDM system parameters

NumSC = 64; % Number of subcarriers
NumPilot = 64; % Number of pilot subcarriers
PilotSpacing = NumSC/NumPilot;
NumPilotSym = 1;
NumDataSym = 1;
NumOFDMsym = NumPilotSym + NumDataSym;

% QPSK modulation
Mod_Constellation = [1-1j;1+1j;-1+1j;-1-1j]; 
NumClass = numel(Mod_Constellation);
Label = 1:NumClass; % Random labels for QPSK symbols

% Pilot symbols - Fixed during the whole transmission
FixedPilot = 1/sqrt(2)*complex(sign(rand(1,NumPilot)-0.5),sign(rand(1,NumPilot)-0.5)); 

%% Channel generation

NumPath = 20;
LengthCP = 16; % The length of the cyclic prefix
% The channel matrix generated using the 3GPP TR38.901 channel model of the
% writer's own implementation, which is saved and loaded:
load('SavedChan.mat'); 

% One can replace the 3GPP channel with the narrowband Rayleigh fading channel:  
%h = 1/sqrt(2)/sqrt(NumPath)*complex(randn(NumPath,1),randn(NumPath,1)); 

H = fft(h,NumSC,1); 

%% SNR calculation

Es_N0_dB = 40;
Es_N0 = 10.^(Es_N0_dB./10);
N0 = 1./Es_N0;
NoiseVar = N0./2;

%% Subcarrier selection

% This is the subcarrier selected for the saved channel h, which can be
% replaced by the following process.
idxSC = 26;

% Select the subcarrier randomly whose gain is above the median value
MedianGain = median(abs(H).^2);
[PossibleSC,~] = find(logical(abs(H).^2 >= MedianGain) == 1);
%idxSC = PossibleSC(randi(length(PossibleSC)));

%% Training data generation

% Training data is generated in the sequence of the modulation
% constellation points.

X = []; % Data
Y = []; % Labels

% Size of dataset to be defined
NumPacketPerClass = 250*1e1; % Number of packets per modulation symbol

% Same pilot sequences used in all packets
FixedPilotAll = repmat(FixedPilot,1,1,NumPacketPerClass); 

% Loop over constellation symbols
for c = 1:NumClass
    
    % OFDM pilot symbol (can be interleaved with random data symbols)
    PilotSym = 1/sqrt(2)*complex(sign(rand(NumPilotSym,NumSC,NumPacketPerClass)-0.5),sign(rand(NumPilotSym,NumSC,NumPacketPerClass)-0.5)); 
    PilotSym(1:PilotSpacing:end) = FixedPilotAll;
    
    % OFDM data symbol
    DataSym = 1/sqrt(2)*complex(sign(rand(NumDataSym,NumSC,NumPacketPerClass)-0.5),sign(rand(NumDataSym,NumSC,NumPacketPerClass)-0.5)); 
    % The data symbol of the current class on the selected subcarrier
    CurrentSym = 1/sqrt(2)*Mod_Constellation(c)*ones(NumDataSym,1,NumPacketPerClass); 
    DataSym(:,idxSC,:) = CurrentSym;
    
    % Transmitted OFDM frame
    TransmittedPacket = [PilotSym;DataSym];

    % Received OFDM frame
    ReceivedPacket = genTransmissionReceptionOFDM(TransmittedPacket,LengthCP,h,NoiseVar);
    
    % Training data collection
    DataLabel = Label(c)*ones(1,NumPacketPerClass); % Data label for the current class on the selected subcarrier
    [feature,label,~] = getFeatureAndLabel(real(ReceivedPacket),imag(ReceivedPacket),DataLabel,c);
    featureVec = mat2cell(feature,size(feature,1),ones(1,size(feature,2))); 
    X = [X featureVec];
    Y = [Y label]; 
    
end

% Re-organize the dataset
X = X.';
Y = Y.';
tempX = [];
tempY = [];
for n = 1:NumPacketPerClass
    tempX = [tempX;X(n:NumPacketPerClass:end)];
    tempY = [tempY;Y(n:NumPacketPerClass:end)];
end

% Split the dataset into training set, validation set and testing set
TrainSize = 4/5;
ValidSize = 1/5;

NumSample = NumClass*NumPacketPerClass;

% Training data
XTrain = tempX(1:NumSample*TrainSize);
YTrain = categorical(tempY(1:NumSample*TrainSize));

XValid = tempX(NumSample*TrainSize+1:end);
YValid = categorical(tempY(NumSample*TrainSize+1:end));

%% Save the training data for training the neural network

save('TrainingData.mat','XTrain','YTrain','NumOFDMsym','NumSC','Label');
save('ValidationData.mat','XValid','YValid');
save('SimParameters.mat','NumPilotSym','NumDataSym','NumSC','idxSC','h','LengthCP','FixedPilot','Mod_Constellation','Label'); 

