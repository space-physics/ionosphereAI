function [outImg,count] = bboxCount(areaBbox,bbox,RGBimg,bboxRatioThres,line_row,count,hshapeBbox,htextCount)

outImg = RGBimg; %copy original RGB image for a final output image

Idx = bbox(:,2) > line_row; % Select boxes which are in the ROI.
    %if ~isempty(bbox), display(bbox), end

    % Based on dimensions, exclude objects which are not feasible. 
    % When the ratio between the area of the blob and the area of the bounding box
    % is above bboxRatioThres, classify it as a desired object.
    ratio = zeros(length(Idx),1);
    ratio(Idx) = single(areaBbox(Idx,1))./single(bbox(Idx,3).*bbox(Idx,4));
    ratiob = ratio > bboxRatioThres;
    countCurr = int32(sum(ratiob));    % Number of detections
    count = count + countCurr;
    bbox(~ratiob,:) = int32(-1);   

    
    outImg(1:15,1:30,:) = 0;  % Black background for displaying count
    outImg = step(hshapeBbox , outImg, bbox); % Draw bounding rectangles around the detection.
    outImg = step(htextCount, outImg, count); % Display the number tracked and a white line showing the ROI.


end %function