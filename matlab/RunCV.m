%% example
% RunCV('~/data/2010-08-03/rx40rx51/fm103.5/ambi',1280875550,1280878249)

function RunCV(datadir,varargin)
%% user parameters

p = inputParser;
addOptional(p,'startUT',nan)
addOptional(p,'stopUT',nan)
addParamValue(p,'bits',16) %#ok<*NVREPL>
parse(p,varargin{:})
U = p.Results;

%% load data
if ~exist('Imgs','var')
try
    display('loading from MAT file')
[Imgs,UP,rangeKM,velocityMPS,utcDN] = ISraw1(datadir,'.h5',U.startUT,U.stopUT,0,0,0,1,100);
catch
    display('saving new MAT file')
    [Imgs,UP,rangeKM,velocityMPS,utcDN] = ISraw1(datadir,'.h5',U.startUT,U.stopUT,0,0,1,0,100);
end
end
%% user parameters
UP.doFrameHist = false;
UP.doWienerFilt = true;
UP.doGMM = true;
UP.plotPP = false; %pixel processes
%_---------------
UP.doWriteVideo = false;
UP.writeVideoFN = 'firstPassDet.avi';
UP.writeVideoFrameRate = 2;
UP.writeVideoQuality = 95;
%=-------------------

UP.RangeMinKM = 400; % 400km per F. Lind for ionosphere at these sites rx40rx51

CP.MinConnected = 100; % minimum number of connected pixels to be considered target
CP.MaxConnected =  400000;
CP.MaxNumBlobs = 5;
CP.ipp = 80:125;
%% do work
% Imgs = graytoUint8(imresize(Imgs,UP.rs),UP.clims);
pp = CVis(Imgs,UP,rangeKM,velocityMPS,utcDN,CP);
end %function