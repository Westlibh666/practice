%% Testing
% 
% This script
%   1. generates testing data for each SNR point;
%   2. calculates the symbol error rate (SER) based on deep learning (DL), 
%   least square (LS) and minimum mean square error (MMSE).

%% Clear workspace

clear variables;
close all;

%% Load common parameters and the trained NN

load('SimParametersPilot64.mat');
load('TrainedNetPilot64.mat');

%% Other simulation parameters

NumPilot = length(FixedPilot);
PilotSpacing = NumSC/NumPilot;
NumOFDMsym = NumPilotSym+NumDataSym;
NumClass = length(Label);
NumPath = length(h);

% Load pre-calculated channel autocorrelation matrix for MMSE estimation
% This autocorrelation matrix is calculated in advance using the 3GPP
% channel model, which can be replaced accordingly.
load('RHH.mat');

%% SNR range

Es_N0_dB = 0:2:20; % Es/N0 in dB
Es_N0 = 10.^(Es_N0_dB./10); % linear Es/N0
N0 = 1./Es_N0;
NoiseVar = N0./2;

%% Testing data size

NumPacket = 10000; % Number of packets simulated per iteration

%% Simulation

% Same pilot sequences used in training and testing stages
FixedPilotAll = repmat(FixedPilot,1,1,NumPacket); 

% Number of Monte-Carlo iterations
NumIter = 1;

% Initialize error rate vectors
SER_DL = zeros(length(NoiseVar),NumIter);
SER_LS = zeros(length(NoiseVar),NumIter);
SER_MMSE = zeros(length(NoiseVar),NumIter);

for i = 1:NumIter
    
    for snr = 1:length(NoiseVar)
        
        %% 1. Testing data generation
        
        noiseVar = NoiseVar(snr);
                
        % OFDM pilot symbol (can be interleaved with random data symbols)
        PilotSym = 1/sqrt(2)*complex(sign(rand(NumPilotSym,NumSC,NumPacket)-0.5),sign(rand(NumPilotSym,NumSC,NumPacket)-0.5)); 
        PilotSym(1:PilotSpacing:end) = FixedPilotAll;
    
        % OFDM data symbol
        DataSym = 1/sqrt(2)*complex(sign(rand(NumDataSym,NumSC,NumPacket)-0.5),sign(rand(NumDataSym,NumSC,NumPacket)-0.5)); 
    
        % Transmitted OFDM frame
        TransmittedPacket = [PilotSym;DataSym];
        
        % Received OFDM frame
        ReceivedPacket = genTransmissionReceptionOFDM(TransmittedPacket,LengthCP,h,noiseVar);
        
        % Collect the data labels for the selected subcarrier
        DataLabel = zeros(size(DataSym(:,idxSC,:)));
        for c = 1:NumClass
            DataLabel(logical(DataSym(:,idxSC,:) == 1/sqrt(2)*Mod_Constellation(c))) = Label(c);
        end
        DataLabel = squeeze(DataLabel); 

        % Testing data collection
        XTest = cell(NumPacket,1);
        YTest = zeros(NumPacket,1);       
        for c = 1:NumClass
            [feature,label,idx] = getFeatureAndLabel(real(ReceivedPacket),imag(ReceivedPacket),DataLabel,Label(c));
            featureVec = mat2cell(feature,size(feature,1),ones(1,size(feature,2))); 
            XTest(idx) = featureVec;
            YTest(idx) = label;
        end
        YTest = categorical(YTest);
        
        %% 2. DL detection
        
        YPred = classify(Net,XTest,'MiniBatchSize',MiniBatchSize);
        SER_DL(snr,i) = 1-sum(YPred == YTest)/NumPacket;
        
        %% 3. LS & MMSE detection
        
        % Channel estimation
        wrapper = @(x,y) performChanEstimation(x,y,RHH,noiseVar,NumPilot,NumSC,NumPath,idxSC);
        ReceivedPilot = mat2cell(ReceivedPacket(1,:,:),1,NumSC,ones(1,NumPacket));
        PilotSeq = mat2cell(FixedPilotAll,1,NumPilot,ones(1,NumPacket));
        [EstChanLS,EstChanMMSE] = cellfun(wrapper,ReceivedPilot,PilotSeq,'UniformOutput',false);
        EstChanLS = cell2mat(squeeze(EstChanLS));
        EstChanMMSE = cell2mat(squeeze(EstChanMMSE));
        
        % Symbol detection
        SER_LS(snr,i) = getSymbolDetection(ReceivedPacket(2,idxSC,:),EstChanLS,Mod_Constellation,Label,DataLabel);
        SER_MMSE(snr,i) = getSymbolDetection(ReceivedPacket(2,idxSC,:),EstChanMMSE,Mod_Constellation,Label,DataLabel);
        
    end
    
end

SER_DL = mean(SER_DL,2).';
SER_LS = mean(SER_LS,2).';
SER_MMSE = mean(SER_MMSE,2).';


figure();
semilogy(Es_N0_dB,SER_DL,'r-o','LineWidth',2,'MarkerSize',10);hold on;
semilogy(Es_N0_dB,SER_LS,'b-o','LineWidth',2,'MarkerSize',10);hold on;
semilogy(Es_N0_dB,SER_MMSE,'k-o','LineWidth',2,'MarkerSize',10);hold off;
legend('Deep learning (DL)','Least square (LS)','Minimum mean square error (MMSE)');
xlabel('Es/N0 (dB)');
ylabel('Symbol error rate (SER)');

%%

function [EstChanLS,EstChanMMSE] = performChanEstimation(ReceivedData,PilotSeq,RHH,NoiseVar,NumPilot,NumSC,NumPath,idxSC)
% This function is to perform LS and MMSE channel estimations using pilot
% symbols, second-order statistics of the channel and noise variance [1].

% [1] O. Edfors, M. Sandell, J. -. van de Beek, S. K. Wilson and 
% P. Ola Borjesson, "OFDM channel estimation by singular value 
% decomposition," VTC, Atlanta, GA, USA, 1996, pp. 923-927 vol.2.


PilotSpacing = NumSC/NumPilot;

%%%%%%%%%%%%%%% LS estimation with interpolation %%%%%%%%%%%%%%%%%%%%%%%%%

H_LS = ReceivedData(1:PilotSpacing:NumSC)./PilotSeq;
H_LS_interp = interp1(1:PilotSpacing:NumSC,H_LS,1:NumSC,'linear','extrap');
H_LS_interp = H_LS_interp.';

%%%%%%%%%%%%%%%% MMSE estimation based on LS %%%%%%%%%%%%%%%%

[U,D,V] = svd(RHH);
d = diag(D);

InvertValue = zeros(NumSC,1);
if NumPilot >= NumPath
    
    InvertValue(1:NumPilot) = d(1:NumPilot)./(d(1:NumPilot)+NoiseVar);
    
else
    
    InvertValue(1:NumPath) = d(1:NumPath)./(d(1:NumPath)+NoiseVar);
    
end

H_MMSE = U*diag(InvertValue)*V'*H_LS_interp;

%%%%%%%%%%%%%%% Channel coefficient on the selected subcarrier %%%%%%%%%%%

EstChanLS = H_LS_interp(idxSC);
EstChanMMSE = H_MMSE(idxSC);

end

%% 

function SER = getSymbolDetection(ReceivedData,EstChan,Mod_Constellation,Label,DataLabel)
% This function is to calculate the symbol error rate from the equalized
% symbols based on hard desicion. 

EstSym = squeeze(ReceivedData)./EstChan;

% Hard decision
DecSym = sign(real(EstSym))+1j*sign(imag(EstSym));
DecLabel = zeros(size(DecSym));
for c = 1:length(Mod_Constellation)
    DecLabel(logical(DecSym == Mod_Constellation(c))) = Label(c);
end

SER = 1-sum(DecLabel == DataLabel)/length(EstSym);

end






