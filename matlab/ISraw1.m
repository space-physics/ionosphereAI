% Takes HDF5 files in a specified directory, loads, displays, and machine vision
% processes them
%
% Michael Hirsch, Boston University
% http://blogs.bu.edu/mhirsch
% Version 0: Nov 2013
% Version 1: Dec 2013
% Tested with Matlab R2013a on Linux
%
%
% Procedure:
% 0) process user input
% 1) Load raw data from HDF5 or MAT files
% 2) Plot raw data
% 3) Machine Vision segmentation 
% 4) Plot Machine Vision results
%
%
% Example for data surrounding Radio Science 2013 Figure 13:
% ISraw1('~/data/2010-08-03/rx40rx51/fm103.5/ambi','.h5',1280877299,1280877579,1,0,0,0)
% 
% save the 2010-Aug-03 data for faster processing
% ISraw1('~/data/2010-08-03/rx40rx51/fm103.5/ambi','.h5',[],[],0,0,1,0)
%------------------------------------------------------------------------------------------
% load the 2010-Aug-03 data for GMM processing (not yet working)
% [Imgs,UP,rangeKM,velocityMPS,utcDNout] = ISraw1('~/data/2010-08-03/rx40rx51/fm103.5/ambi','.h5',[],[],0,0,0,1);
% Imgs = graytoUint8(imresize(Imgs,UP.rs),UP.clims);
% GMMis(Imgs,UP,rangeKM,velocityMPS,utcDNout);
%------------------------------------------------------------
% load the 2010-Aug-03 data for CV processing (work in progress)
% [Imgs,UP,rangeKM,velocityMPS,utcDNout] = ISraw1('~/data/2010-08-03/rx40rx51/fm103.5/ambi','.h5',[],[],0,0,0,1);
% Imgs = graytoUint8(imresize(Imgs,UP.rs),UP.clims);
% CVis(Imgs,UP,rangeKM,velocityMPS,utcDNout);
%
% you can obtain the necessary data for this example event by typing in
% Terminal:
% mkdir -p ~/data
% wget --quota=10000m -r -np -nc -nH --cut-dirs=3 --random-wait --wait 0.5 --reject index.html* -e robots=off -P ~/data/ http://atlas.haystack.mit.edu/isis/fermi/events/2010-08-03/rx40rx51/
% wget --quota=10000m -r -np -nc -nH --cut-dirs=3 --random-wait --wait 0.5 --reject index.html* -e robots=off -P ~/data/ http://atlas.haystack.mit.edu/isis/fermi/events/2010-08-13/rx40rx51/
% wget --quota=10000m -r -np -nc -nH --cut-dirs=3 --random-wait --wait 0.5 --reject index.html* -e robots=off -P ~/data/ http://atlas.haystack.mit.edu/isis/fermi/events/2010-08-05/rx40rx51/
% wget --quota=10000m -r -np -nc -nH --cut-dirs=3 --random-wait --wait 0.5 --reject index.html* -e robots=off -P ~/data/ http://atlas.haystack.mit.edu/isis/fermi/events/2010-08-04/rx40rx51/
%
function [Imgs,UP,rangeKM,velocityMPS,utcDN,filenames] = ISraw1(dataDir,dataExt,startUTCsec,stopUTCsec,...
                                  plotRaw,plotMeans,saveH5,loadH5,RangeMinKM,doWienerFilt,doGMM,diagGMM)

clc
%% Step 0: user parameters
if nargin<1 || isempty(dataDir)
    %dataDir = '~/data/annotated_data'; 
    %dataDir = '~/data/2010-08-03/rx40rx51/fm103.5/ambi';  %ionosphere + aircraft
    %dataDir = '~/data/2010-08-03/rx40rx51/fm103.5/ambi'; 
    %dataDir = '~/data/2010-08-13/rx40rx51/fm103.5/ambi'; 
    %dataDir = '~/data/2010-08-05/rx40rx51/fm103.5/ambi';
    dataDir =  '~/data/2013-11-20/rx50rx50/ambi';
end

if nargin<2 || isempty(dataExt), dataExt = '.h5'; end

%will show frames back to Unix UTC epoch start (Jan 1 1970)
if nargin<3 || isempty(startUTCsec), UP.startUTCsec = 0; else UP.startUTCsec = startUTCsec; end 

%will show frames up to 32-bit Unix UTC epoch end (Jan 19 2038) 
if nargin<4 || isempty(stopUTCsec), UP.stopUTCsec = 2^31 - 1; else UP.stopUTCsec = stopUTCsec; end 

if nargin<5, UP.plotRaw = true; else UP.plotRaw = plotRaw; end
if nargin<6, UP.plotMeans = true; else UP.plotMeans = plotMeans; end

if nargin<7, UP.saveH5 = false; else UP.saveH5 = saveH5; end
if nargin<8, UP.loadH5 = false; else UP.loadH5 = loadH5; end

if nargin<9, UP.RangeMinKM = 100; else UP.RangeMinKM = RangeMinKM; end

if nargin<10, UP.doWienerFilt = false; else UP.doWienerFilt = doWienerFilt; end
if nargin<11, UP.doGMM = false; else UP.doGMM = doGMM; end
if nargin<12, UP.diagGMM = true; else UP.diagGMM = diagGMM; end

[UP, filenames,nRawFiles,meanRaw,utcDN,hg,Imgs] = userParams(dataDir,dataExt,UP);
h5fn = [dataDir,filesep,UP.dirQual,UP.fnStem,'.mat'];


%% Step 1a: compute parameters
if ~UP.loadH5
    
display(['Using filename regexp: ',UP.utcRegExp,' on ',int2str(nRawFiles),' files.'])
j=0; 
for i = 1:nRawFiles
try %not the most efficient placing, but we want to keep going with next file on failure
%% Step 1b: Load data
    fn = filenames(i).name;
    j = j+1;

    fn = [dataDir,filesep,fn]; %#ok<AGROW>

[rangeKM,velocityMPS,SCRdb,currUTC,integration_time,logamb] = getFrame(fn,dataExt);

if isempty(UP.clims0) %auto frame-by-frame contrast
    clims = [min(SCRdb(:)),max(SCRdb(:))];
else %user-specified clims
    clims = UP.clims0;
end
%% Step 2: plot raw data
if UP.plotRaw
%     figure(1)
%     imagesc(rangeKM,velocityMPS,logamb)
%         set(gca,'ydir','normal')
%     colorbar 
%     xlabel('Range [km]')
%     ylabel('Velocity [m/s]')
    set(hg.trw,'string',['SCR: ',datestr(currUTC)])
    set(hg.imrw,'cdata',SCRdb)
end %if plotRaw
%% output variables
Imgs(:,:,j) = SCRdb; 

if UP.plotMeans
meanRaw(j,1) = mean(logamb(:));
meanRaw(j,2) = median(logamb(:));

set(hg.pmn,'xdata',currUTC,'ydata',meanRaw(:,1)) %mean
set(hg.pmd,'xdata',currUTC,'ydata',meanRaw(:,2)) %median
end %if

%figure(99),imagesc(normImgs(:,:,j)),colorbar
 drawnow %necessary to update each looparound
if ~mod(i,50), 
    display(['File ',int2str(i),'/',int2str(nRawFiles)])
end

catch err 
    fprintf(['In file ',fn,' '])
      % display(['raw plot failure on file: ',fn,'  i=',int2str(i),'  j=',int2str(j)]) 
       display(getReport(err,'extended'))
       %display([err.identifier,': ',err.message])
end %try
    
end % for

UP.clims = clims; %in case they were automatically set %FIXME

if UP.saveH5
    display(['saving to MAT file: ',h5fn])
    save(h5fn,'Imgs','rangeKM','velocityMPS','UP','utcDN')   
end %if saveH5

%% cleanup


if UP.plotMeans
   figure(hg.fmn)
   datetick
   title({[datestr(utcDN(1),'yyyy-mmm-dd'),' to ',datestr(utcDN(end),'yyyy-mmm-dd'),', int_time: ',num2str(integration_time),' sec.']},'interpreter','none')
   grid on
end

if UP.plotRaw
   figure(hg.frw)
   title({[datestr(utcDN(1),'yyyy-mmm-dd'),' to ',datestr(utcDN(end),'yyyy-mmm-dd'),', int_time: ',num2str(integration_time),' sec.']},'interpreter','none')
end 
else % load from HDF5 file
    display(['Loading prestored data from: ',h5fn])
   [Imgs,rangeKM,velocityMPS,UP,utcDN] = getH5(h5fn,UP);
end %if loadH5

%% truncate range extent
display(['Using Minimum Range of ',num2str(UP.RangeMinKM),' km.'])
BadRangeInd = rangeKM < UP.RangeMinKM;
Imgs(:,BadRangeInd,:) = [];
rangeKM(BadRangeInd) = [];

%% Step 3: Machine Vision process



if UP.doGMM && exist('GMMis.m','file') 
    Imgs = graytoUint8(imresize(Imgs,UP.rs),UP.clims);
    GMMis(Imgs,UP,rangeKM,velocityMPS,utcDN);
else
    display('Skipping Machine Vision processing')
end

if nargout==0, clear, end
end %function

function [UP, files,nRawFiles,meanRaw,utcDN,hg,Imgs] = userParams(dataDir,dataExt,UP)
hg = [];


%------- highlight a pixel for GMM diagnosis as specified here ---
UP.pixRangeKM =     200;
UP.pixVelocityMPS = -200;
%-----------------------------------
UP.rs = 1; %scaling image size
%---------- end --------------

%----------- regexp -------------
dirQualRegExp = '(?<=.*/)\d{4}-\d{2}-\d{2}(?=/.*)'; %gets just the date

dirQual = regexp(dataDir,dirQualRegExp,'match','once'); 
if isempty(dirQual)
    if ~isempty(regexp(dataDir,'annotated_data', 'once'))
        dirQual = 'annotated_data';
    else
        error(['no match for data directory',dataDir])
    end
end
switch dirQual
    case {'2010-08-03','annotated_data'}
     UP.utcRegExp = '(?<=.*@)\d{10}(?=.h5)'; 
     UP.fnStem = 'iax001_000';
     UP.clims0 = [0,6]; 
    case {'2010-08-13','2010-08-05/'}
     UP.utcRegExp = '(?<=.*_)\d{10}(?=_\d{3}.h5)'; 
     UP.fnStem = 'iax_001';
     UP.clims0 = [10,12.5]; 
    case {'2013-11-20'}
     UP.utcRegExp = '(?<=.*@)\d{10}(?=.h5)'; 
     UP.fnStem = 'iax001_001';
     UP.clims0 = [];
    otherwise, error(' sorry I don''t have the regexp for this date')
end
UP.dirQual = dirQual;

%% get directory info
display(['Listing directories in ',dataDir])
fullDir = [dataDir,filesep,UP.fnStem,'*',dataExt];
files = dir(fullDir);
filenames = {files.name};

%% filter by UTC sec user criteria
utcfn = str2double(regexp(filenames,UP.utcRegExp,'match','once')); utcfn = utcfn(:);
badFNind = utcfn < UP.startUTCsec | utcfn > UP.stopUTCsec;

files(badFNind) = []; %qualify by start/stop times
utcfn(badFNind) = [];
nRawFiles = length(files);

utcDN = datenum([repmat([1970, 1, 1, 0, 0],[nRawFiles,1]), utcfn]);

if nRawFiles==0
    error(['I didn''t find any suitable files in your time range in directory/stem: ',dataDir,filesep,UP.fnStem])
end % if nRawFIles==0

display(['Using data from: ',datestr(utcDN(1),'yyyy-mmm-ddTHH:MM:ss'),' to ',datestr(utcDN(end),'yyyy-mmm-ddTHH:MM:ss')])


if ~UP.loadH5
%% priming read
% assuming all parameters stay the same for a given directory/fnStem pair
fn = [dataDir,filesep,files(1).name];
display(['Priming read from: ',fn])

[rangeKM,velocityMPS] = getFrame(fn,dataExt);

UP.nRow = length(velocityMPS);
UP.nCol = length(rangeKM);

MaxRamBytes = 2e9; % to avoid swap crashing
RamUsedImgs = ( UP.nRow * UP.nCol * nRawFiles * 64 / 8 );

if RamUsedImgs > MaxRamBytes
    error(['Loading ',int2str(nRawFiles),' files, at ',num2str(RamUsedImgs/1e9,'%0.1f'),' GB into RAM is excessive. Try limiting number of files loaded by Start/Stop UTC times'])
end

%------- set some nuisance parameters -------------
meanRaw = nan(nRawFiles,2); % at most
Imgs = nan(UP.nRow,UP.nCol,nRawFiles);
%--------------------------------------------------


%% plot setup
if UP.plotRaw
    hg.frw = figure(1);
    if isempty(UP.clims0) %auto frame-by-frame contrast
        hg.imrw = imagesc(rangeKM,velocityMPS,nan(UP.nRow,UP.nCol));
    else %specified fixed contrast
        hg.imrw = imagesc(rangeKM,velocityMPS,nan(UP.nCol,UP.nRow),UP.clims0);
    end
    set(gca,'ydir','normal')
    colorbar 
    xlabel('Range [km]')
    ylabel('Velocity [m/s]')
    hg.trw = title('');
end

if UP.plotMeans
hg.fmn = figure(33); clf(33)
hg.pmn = line(0,0,'marker','.','markeredgecolor','b','displayname','mean','linestyle','none');
hg.pmd = line(0,0,'marker','.','markeredgecolor','r','displayname','median','linestyle','none');

xlabel('UTC time')
ylabel('relative value')
legend('show','location','best')
title('Mean and Median of log10(ambiguity)')
end 
else
    meanRaw =[];
    utcDN = [];
    Imgs = [];
end %if ~UP.loadH5

end %function

function [rangeKM,velocityMPS,SCRdb,currUTC,integration_time,logamb] = getFrame(fn,dataExt)

switch dataExt
    case '.mat', load(fn) %typically only .h5 files are available
    case '.h5',  [ambiguity,rangeKM,velocityMPS,t_utc,integration_time] = readFMh5(fn);
    otherwise, error(['unknown data type ',dataExt])
end %switch

logamb = log10(abs(ambiguity));

%logamb = log10(abs(ambiguity)); %for plotting in imagesc
SCRdb = 10*log10(abs(ambiguity./median(ambiguity(:))));

currUTC = datetime(1970, 1, 1, 0, 0, t_utc); %Unix epoch Jan 1,1970

end

function [Imgs,rangeKM,velocityMPS,UP,utcDNout] = getH5(h5fn,UP) %#ok<STOUT>
%protect the main namespace for UP
UPold = UP;

load(h5fn,'Imgs','rangeKM','velocityMPS','UP','utcDNout')

UP.doWienerFilt = UPold.doWienerFilt;
UP.doGMM = UPold.doGMM;
UP.diagGMM = UPold.diagGMM;
end
