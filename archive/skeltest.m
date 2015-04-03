% tested on Matlab R2013b
% Michael Hirsch
%
%
function skeltest(vidFN)

%% user parameters
if nargin<1, 
    %vidFN = [getenv('HOME'),'/Prospectus/vid/X1387_032307_112005.36_full.avi']; 
    vidFN = 'horizslide-vertbar-.avi';  %generate by RunGenOFtestPattern(0,'avi','horizslide','vertbar',128,256,256,0.5,0.5,1,8,5,3,35);
end

intThres = 0.22; %arbitrary, assumed video intensities \in [0,1]
nSkel = 8; %arbitrary, number of skeleton iterations
%% setup cv objects
hv = VideoReader(vidFN); %just to get file parameters 
nRow = get(hv,'Height');
nCol = get(hv,'Width');

hv = vision.VideoFileReader(vidFN,'ImageColorSpace','Intensity');
%% initialization
thresImg = false(nRow,nCol);
%% setup plots
figure(1); clf(1)
hthr = image([1 nCol],[1 nRow],thresImg,'cdatamapping','scaled');
title('raw Image')
colormap('gray')
colorbar

figure(2);clf(2);
hsk = image([1 nCol],[1 nRow],thresImg,'cdatamapping','scaled');
title('Skeleton')
colormap('gray')
colorbar

hf = figure(3);clf(3);
hep = image([1 nCol],[1 nRow],thresImg,'cdatamapping','scaled');
title('Endpoints')
colormap('gray')
colorbar
%% do work
while ~isDone(hv)
    
    frame = step(hv); %get grayscale video frame
    
    thresImg = false(nRow,nCol);
    thresImg(frame>intThres) = true; %create binary thresholded image %there are better ways to do this such as Otsu etc.
    set(hthr,'cdata',thresImg)
    
    skelImg = bwmorph(thresImg,'skel',nSkel);
    set(hsk,'cdata',skelImg)
    
    brnpts = bwmorph(skelImg,'branchpoints');
    endpts = bwmorph(skelImg,'endpoints');
    
    
    
    drawnow
    pause(0.01)
    
end %while
close(hv)
end