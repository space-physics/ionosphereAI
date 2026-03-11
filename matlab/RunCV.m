%% example
% RunCV('~/data/2010-08-03/rx40rx51/fm103.5/ambi',1280875550,1280878249)

function pp = RunCV(datadir, startUT, stopUT, bits)
arguments
    datadir (1,1) string
    startUT (1,1) datetime
    stopUT (1,1) datetime
    bits (1,1) {mustBeInteger} = 16
end

% load data
try
  disp('loading from MAT file')
  [Imgs,UP,rangeKM,velocityMPS,utcDN] = ISraw1(datadir,'.h5',startUT,stopUT,0,0,0,1,100);
catch
  disp('saving new MAT file')
  [Imgs,UP,rangeKM,velocityMPS,utcDN] = ISraw1(datadir,'.h5',startUT,stopUT,0,0,1,0,100);
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

end
