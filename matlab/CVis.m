function pp = CVis(Imgs,UP,rangeKM,velocityMPS,utcDN,CP) %#ok<*INUSD>
% by Michael Hirsch Dec 2013 mhirsch@bu.edu
%
% a playground for preliminary computer vision techniques using MIT haystack passive fm data.

%% the simple things first
% 
% # ROI: Commercial airliners < 15km ALTITUDE;  Ionosphere > 90 km ALTITUDE
% problem--we get slant range with typical two-site configuration. Can 3D position be obtained with multiple
% sites e.g. 4 sites? Hard to get adequate SCR at all 4 sites?
% # INTENSITY(SCR): would have to be in conjunction with slant range
% 
%% Consider aircraft
% For a two-site configuration where only 
% 
% * Slant Range
% * Doppler
% 
% are available, how can we qualify aircraft?
% how does radio horizon impact max range (must be in common view of TX and
% both RX) when considering maximum 15km altitude of aircraft.
%% Consider Ionosphere
% Ionosphere minimum 90km altitude (probably couple hundred km for the
% observable scattering type)
%
%% for this assigment, need to know:
% 
% # what governs viewing geometry limits
% # what kind of background (clutter) variations must be tolerated
% # what is a tolerable minimum SCR for XX% $P_d$ vs $P_{fa}$
%% user parameters
CP.HistXlim = [-10, 10]; %arbitrary, x-limits for histogram
CP.HistYlim = [0, 10^6]; %arbitrary, y-limits for histogram
CP.HistXbin = 64; %arbitrary, number of histogram bins

CP.ppRangeVel = [350 -500;
                     400 350]; %arbitrary, range over which to compute pmf (prob. mass fcn)
CP.Npp = size(CP.ppRangeVel,1); % arbitrary, number of points in pmf

NoisePwrEst = []; % [] means wiener2 will estimate the noise power itself for each frame


%% compute parameters
display(['Using only hits with more than ',int2str(CP.MinConnected),' pixels.'])
% check if user has requested smaller minimum data extents and trim data
% accordingly
display(['Trimming data to Minimum Range of ',num2str(UP.RangeMinKM),' km.'])
BadRangeInd = rangeKM < UP.RangeMinKM;
Imgs(:,BadRangeInd,:) = [];
rangeKM(BadRangeInd) = [];

[nRow,nCol,nFrames] = size(Imgs);

for ipp = 1:CP.Npp
    ppInd(ipp,1) = round(interp1(rangeKM,1:nCol,...
                               CP.ppRangeVel(ipp,1),'linear')); %#ok<AGROW>
    ppInd(ipp,2) = round(interp1(velocityMPS,1:nRow,...
                               CP.ppRangeVel(ipp,2),'linear')); %#ok<AGROW>
end %for
pp = NaN(nFrames,CP.Npp);
NoisePwr = NaN(nFrames,1);

%% Step 0: Histogram of SCR
% let's get a sense of what relative intensities we're dealing with.


%get date for each frame
utcDS = datestr(utcDN);

h = makeFigs(Imgs,rangeKM,velocityMPS,UP,CP,nFrames);
hJunk = [];
for iFrm = 1:nFrames-1 %80:150
    
    currImg = Imgs(:,:,iFrm);
    
    if UP.doWienerFilt
     [currImg, NoisePwr(iFrm)] = wiener2(currImg,[5,5],NoisePwrEst); % 2D Wiener filter (reduce noise)
    end
    
% update raw img
    set(h.ir,'cdata',currImg)
    set(h.tr,'string',['SCR [dB] @ ',utcDS(iFrm,:),' UTC. Detections outlined in Green'])
    
% show per-frame histogram
if UP.doFrameHist
     [nhist,histcent] = hist(currImg(:),CP.HistXbin);
     set(h.bh,'xdata',histcent,'ydata',nhist)
end

% plot pixel process
if UP.plotPP
    for ipp = 1:CP.Npp
        pp(iFrm,ipp) = currImg(ppInd(ipp,1),ppInd(ipp,2));
        set(h.ppp(ipp),'ydata',pp(:,ipp))
    end %for Npp
end

% GMM: Detect foreground pixels
if UP.doGMM
%% point detection
% FIXME: should be its own option
    
%% GMM
     GMMfg = step(h.FGdet, currImg); 
     set(h.ifg,'cdata',GMMfg)
%% morphological
%GMMfg = step(h.MorphOpen,GMMfg);  % morphological open
%GMMfg = step(h.MorphClose,GMMfg); % morphological close
GMMfg = imerode(GMMfg,strel('disk',3));  % morphological erosion  (eliminate isolated pixels)
set(h.iErosion,'cdata',GMMfg) %update erosion display

GMMfg = imdilate(GMMfg,strel('disk',5));  % morphological dilation (gap fill)
set(h.iThrGMMcc,'cdata',GMMfg)    % update dilation display
%-----------------------------------------
%% blob analysis
[bArea,bCentroid,bbox] = step(h.blob,GMMfg);
Ncc = size(bbox,1); % Nx4
delete(hJunk)
hold(h.ar,'on')
hJunk = nan(Ncc,1);
bbox = double(bbox); % for interp1 compatibility
for icc = 1:Ncc
    %have to convert from raw index to physical units (ugh!)
    % (3) and (4) count on uniform spacing of Range,Velocity
   bbi = [round(interp1(1:nCol,rangeKM,bbox(icc,1),'linear')),...
          round(interp1(1:nRow,velocityMPS,bbox(icc,2),'linear')),...
          round( (rangeKM(2) - rangeKM(1)) * bbox(icc,3) ),...
          round( (velocityMPS(2) - velocityMPS(1)) * bbox(icc,4) )];
   hJunk(icc) =  rectangle('position',bbi,'edgecolor','g','parent',h.ar);
end
hold(h.ar,'off')


%% connected component thresholding
%{
GMMccTF = false(nRow,nCol); %used for displaying connected component result
 GMMcc = bwconncomp(GMMfg);
 GMMccPix = GMMcc.PixelIdxList; % list of 8-connected pixels
 numGMMcc = cellfun(@numel,GMMccPix); %number of 8-connected pixels at each pixel
 GMMccPix = GMMccPix(numGMMcc >= CP.MinConnected); % thresholded for at least X connected pixels
 nGoodGMMcc = length(GMMccPix); % number of remaining connected components
 %display connected component result
    for icc = 1:nGoodGMMcc
        GMMccTF(GMMccPix{icc}) = true;
    end %for

  set(h.iThrGMMcc,'cdata',GMMccTF)
%}


end % if doGMM
    drawnow
    
    %pause
    
   if UP.doWriteVideo
       vwFrame = getframe(h.fr);
       writeVideo(h.vidWriter,vwFrame);
   end %if write video
    
   if ~mod(iFrm,25), display(['Frame ',int2str(iFrm),'/',int2str(nFrames)]), end
end %for

if UP.doWriteVideo, close(h.vidWriter); end

%% Statistics of Pixel Processes

%estimate pmf
if UP.plotPP
figure(20),clf(20)
for ipp = 1:CP.Npp
    subplot(CP.Npp,1,ipp)
    [ppBinVal,ppBinCent] = hist(pp(:,ipp),NbinPMF);
    pmf = ppBinVal / sum(ppBinVal);
    stairs(ppBinCent,pmf)   
    ylabel('pmf estimate')
    title(['R = ',num2str(CP.ppRangeVel(ipp,1)),' km, V = ',num2str(CP.ppRangeVel(ipp,2)),' m/s'])
end %for
xlabel('SCR [dB]')
end

% show Wiener2 power estimates
figure(21),clf(21)
plot(1:nFrames,NoisePwr)
xlabel('Frame #')
ylabel('Est. Noise Pwr [dB]')

end %function

function h = makeFigs(Imgs,rangeKM,velocityMPS,UP,CP,nFrames)
display('Creating Vision Toolbox Objects')
%% handle for VideoWriter

if UP.doWriteVideo
   h.vidWriter = VideoWriter(UP.writeVideoFN,'Motion JPEG AVI');
   h.vidWriter.FrameRate = UP.writeVideoFrameRate;
   h.vidWriter.Quality =   UP.writeVideoQuality;
   open(h.vidWriter);
else
   h.vidWriter = [];    
end

%% setup point detector
h.kernLaplace = fspecial('laplacian',0.2);

%% setup GMM

% Create a System object to detect foreground using Gaussian Mixture Models.
if UP.doGMM
  % GMM
  h.FGdet = vision.ForegroundDetector(... 
        'AdaptLearningRate',true,...
        'LearningRate',0.01,... %alpha
        'NumTrainingFrames', 8, ...     % only 3 because of short video
        'InitialVariance', (30/255)^2,... % initial standard deviation of 30/255
        'MinimumBackgroundRatio',0.9,...
        'NumGaussians',3);
  
  %Morphological opening
  %h.MorphOpen = vision.MorphologicalOpen('Neighborhood', strel('disk',2));
  %Morphological closing
  %h.MorphClose = vision.MorphologicalClose('Neighborhood', strel('disk',5));  
  
  %blob (connected components) analysis
  h.blob = vision.BlobAnalysis( ...
                    'CentroidOutputPort', true, ...    % centroid of blob
                    'AreaOutputPort', true, ...        % area of blob
                    'BoundingBoxOutputPort', true, ... % box of blob
                    'OutputDataType', 'double', ...
                    'Connectivity',8,...
                    'MinimumBlobArea', CP.MinConnected, ...
                    'MaximumBlobArea', CP.MaxConnected, ...
                    'MaximumCount', CP.MaxNumBlobs,...      % max # of blobs to compute
                    'ExcludeBorderBlobs',false); % don't allow blobs with 1 or more pixels on border
    
end


%% setup plots
display('initializing plots')
% priming read
currImg = Imgs(:,:,1); 

% raw image figure
h.fr = figure(1); clf(1)
curp = get(gcf,'pos'); set(gcf,'pos',[curp(1),curp(2),520,640])
h.ar = axes('parent',h.fr);
h.ir = imagesc(rangeKM,velocityMPS,currImg,UP.clims);
h.tr = title(h.ar,'');
xlabel('range [km]'), ylabel('velocity [m/s]')
set(h.ar,'ydir','normal')
colormap(h.ar,'jet')
colorbar('peer',h.ar)
%{
cmap = colormap('jet');
cmap(end,:) = 1; %sets last color to white
colormap(h.ar,cmap)
%}

% histogram figure
if UP.doFrameHist
h.fh = figure(2); clf(2)
curp = get(gcf,'pos'); set(gcf,'pos',[curp(1),curp(2),520,460])
h.ah = axes('parent',h.fh);
 [nhist,histcent] = hist(currImg(:),CP.HistXbin);
 h.bh = stairs(histcent,nhist,'parent',h.ah);
 set(h.ah,'yscale','log','xlim',CP.HistXlim,'ylim',CP.HistYlim)
end %if
 

 
% GMM foreground figure
if UP.doGMM

    h.ffg = figure(5); clf(5)
    curp = get(gcf,'pos'); set(gcf,'pos',[curp(1),curp(2),520,640])
    h.afg = axes('parent',h.ffg);
    h.ifg = imagesc(rangeKM,velocityMPS,logical(currImg),[0 1]);
    xlabel('range [km]'), ylabel('velocity [m/s]')
    title('GMM result: Foreground Pixels')
    set(h.afg,'ydir','normal')
    colormap('gray')

    h.fErosion = figure(7);clf(7)
    curp = get(gcf,'pos'); set(gcf,'pos',[curp(1),curp(2),520,640])
    h.aErosion = axes('parent',h.fErosion);
    h.iErosion = imagesc(rangeKM,velocityMPS,logical(currImg),[0 1]);
    xlabel('range [km]'), ylabel('velocity [m/s]')
    title('Morphological Erosion')
    set(h.aErosion,'ydir','normal')
    colormap('gray')

    
    h.fThrGMMcc = figure(6); clf(6)
    curp = get(gcf,'pos'); set(gcf,'pos',[curp(1),curp(2),520,640])
    h.aThrGMMcc = axes('parent',h.fThrGMMcc);
    h.iThrGMMcc = imagesc(rangeKM,velocityMPS,logical(currImg),[0 1]);
    xlabel('range [km]'), ylabel('velocity [m/s]')
    %title(['Connected Components >',int2str(CP.MinConnected)])
    title('Morphological Dilation')
    set(h.aThrGMMcc,'ydir','normal')
    colormap('gray')

end %if doGMM


% plot individual pixel(s)
if UP.plotPP
h.fppp = figure(3); clf(3)
curp = get(gcf,'pos'); set(gcf,'pos',[curp(1),curp(2),520,460])

for ipp = 1:CP.Npp
h.app(ipp) = subplot(CP.Npp,1,ipp);
h.ppp(ipp) = plot(1:nFrames,nan(nFrames,1));
title(['R = ',num2str(CP.ppRangeVel(ipp,1)),' km, V = ',num2str(CP.ppRangeVel(ipp,2)),' m/s'])
%show pixel location on raw figure
line(CP.ppRangeVel(ipp,1),CP.ppRangeVel(ipp,2),'parent',h.ar,'Marker','*','MarkerEdgeColor','g')
%show pixel location of GMM fg figure
if UP.doGMM
 line(CP.ppRangeVel(ipp,1),CP.ppRangeVel(ipp,2),'parent',h.afg,'Marker','*','MarkerEdgeColor','g')
end %if
ylabel('SCR [dB]')
end %for
xlabel('Frame #')
set(h.app,'xlim',[1,nFrames])
end


end
