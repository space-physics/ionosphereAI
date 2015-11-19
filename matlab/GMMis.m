% tracking using Gaussian Mixture Models
% michael hirsch
% for Passive FM radar MIT Haystack project
% Nov 2013
function GMMis(normImgs,UP,rangeKM,velocityMPS,utcDN) %#ok<*INUSD>
%% user parameters

if nargin<2, 
    UP.doWienerFilt = false;
end



%video parameters
xincr = 1320;
yincr = 620;
xsize = 1250 * UP.rs;
ysize = 560 *  UP.rs;


%% setup Machine Vision methods
% get number of frames in video
[nRow,nCol,nFrame] = size(normImgs);
display(['Video has ',int2str(nFrame),' frames, and ',...
    int2str(nCol),'x',int2str(nRow),' pixels.'])


% Create a System object to detect foreground using gaussian mixture models.
hfdet = vision.ForegroundDetector(...
        'AdaptLearningRate',true,...
        'LearningRate',0.01,... %alpha
        'NumTrainingFrames', 5, ...     % only 3 because of short video
        'InitialVariance', (30/255)^2,... % initial standard deviation of 30/255
        'MinimumBackgroundRatio',0.9,...
        'NumGaussians',5);
        
% user tunes to fit arc sizes on video
% e.g. 'MinimumBlobArea', 500, 'MaximumBlobArea', 8600, 'MaximumCount', 80
hblob = vision.BlobAnalysis( ...
                    'CentroidOutputPort', false, ...
                    'AreaOutputPort', true, ...
                    'BoundingBoxOutputPort', true, ...
                    'OutputDataType', 'single', ...
                    'Connectivity',8,...
                    'MinimumBlobArea', 500, ...
                    'MaximumBlobArea', 8600, ...
                    'MaximumCount', 20,...
                    'ExcludeBorderBlobs',true);
                
% for taking mean of frame --> TODO BUG constantly outputting zero !?
% hmean = vision.Mean('RunningMean',false,...
%                     'Dimension','all',...
%                     'ROIProcessing',false);
                
%make green bounding box handle (for final output)
hshapeBbox = vision.ShapeInserter('BorderColor', 'Custom', ...
            'CustomBorderColor', [0 255 0]);

% text handle to count number of "hits"                     
htextCount = vision.TextInserter('Text', '%4d', 'Location',  [1 1], ...
                               'Color', [1 1 1], 'FontSize', 12);

%% make figures
sz = get(0,'ScreenSize');

pos = [20 sz(4)-ysize xsize ysize];
%hVOrig = vision.VideoPlayer('Name', '(0) Original Data', 'Position', pos);
hf(2) = figure(2);
set(hf(2),'pos',pos,'name','(0) Original Data')
hi(2) = imagesc(rangeKM,velocityMPS,nan(nRow,nCol),UP.clims);
line(UP.pixRangeKM,UP.pixVelocityMPS,'marker','*','markerEdgeColor','g','displayName','plottedPoint') % this is the plotted pixel
xlabel('range'),ylabel('velocity')

pos(2) = pos(2)-yincr; % move the next viewer to the right
hVFg = vision.VideoPlayer('Name', '(1) GMM: Foreground', 'Position', pos);

pos(1) = pos(1)+xincr;
hVRes = vision.VideoPlayer('Name', '(2) GMM Results', 'Position', pos);

% if UP.diagGMM
%     diagPixVal = nan(nFrame,1);
%     meanPixVal = diagPixVal;
%    figure(50),clf(50)
%    %cp = get(50,'pos'); cp(3) = cp(3) + 200;
%    %set(50,'pos',cp)
%    
%   % ylims = [11 11.5];%UP.clims;%[55 75];
%    
%    haxpix(1) = axes('parent',50); %axes('parent',50,'units','normalized','pos',[0.075 0.075 0.405 0.85],'ylim',ylims);
%    hppix = line(utcDN,diagPixVal,'parent',haxpix(1),'marker','.','markeredgecolor','r','displayName','(200,-200)');
%    hpavg = line(utcDN,meanPixVal,'parent',haxpix(1),'color','k','linewidth',2,'marker','.','markeredgecolor','r','displayName','mean');
%    xlabel('UTC time')
%    ylabel('Pixel Value (log10(ambg))')
%    title(haxpix(1),['pixel (range,velocity) = (',int2str(UP.pixRangeKM),',',int2str(UP.pixVelocityMPS),'), on ',datestr(utcDN(1),'yyyy-mm-dd')])
%    %find pixel indicies
%    UP.pixRow = findnearest(UP.pixVelocityMPS, velocityMPS);
%    UP.pixCol = findnearest(UP.pixRangeKM,     rangeKM);
%    display(['plotting pixel values from (row,col)=(',int2str(UP.pixRow),',',int2str(UP.pixCol),')'])
%    grid on
%    datetick
%    
%    %haxpix(2) = axes('parent',50,'units','normalized','pos',[0.575 0.075 0.405 0.85],'ylim',ylims);
%    %hpavg = line(utcDN,meanPixVal,'parent',haxpix(1),'marker','.','markeredgecolor','r');
%    %xlabel('UTC time')
%    %ylabel('mean(log10(ambg))')
%    %title(haxpix(2),['Mean(log10(ambg)), frame by frame, on ',datestr(utcDN(1),'yyyy-mm-dd')])
%    %grid on
%    %datetick
% end
 
%% user filtering parameters
line_row = 0; % Detects must be below this y-pixel
bboxRatioThres = 0;

%% GMM Loop

 count = 0;
for i = 1:nFrame
    
%    rawImg = Imgs(:,:,i);
    
    grayImg = normImgs(:,:,i);
    RGBimg = gray2rgb(grayImg);

%     if UP.diagGMM
%         %diagPixVal(i) = grayImg(UP.pixRow,UP.pixCol);
%         diagPixVal(i) = rawImg(UP.pixRow,UP.pixCol);
%         set(hppix,'ydata',diagPixVal)
%         
%         meanPixVal(i) = mean(rawImg(:));
%         %meanPixVal(i) = mean(grayImg(:));
%         %step(hmean,grayImg);
%         set(hpavg,'ydata',meanPixVal)
%     end        
    
    if UP.doWienerFilt
     grayImg = wiener2(grayImg,[8,8]); % (1.0) 2D Wiener filter (reduce noise)
    end

    fg_image = step(hfdet, grayImg); % (2.0) GMM: Detect foreground pixels

    % Estimate the area and bounding box of the blobs in the foreground image.
    [areaBbox, bbox] = step(hblob, fg_image);

    [outImg, count] = bboxCount(areaBbox,bbox,RGBimg,bboxRatioThres,line_row,...
                     count,hshapeBbox,htextCount); %count bounding boxes and  display
    
  %  step(hVOrig, RGBimg);      % Original video
    set(hi(2),'cdata',RGBimg)
    
    step(hVFg,   fg_image);    % Foreground
    step(hVRes,  outImg);      % Bounding boxes around objects

%pause
end

%release(hVOrig)
release(hVFg)
release(hVRes)

end %function
