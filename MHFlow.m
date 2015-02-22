function MHFlow(data)


%data = rawDMCreader(NewVid);
[r,c,nFrame] = size(data);
clim = [1000,4000];
hsvimg(:,:,2) = ones(r,c);
hsvimg(:,:,3) = 0;
gray1 = normFrame(data(:,:,1),16,clim);
hof = vision.OpticalFlow('ReferenceFrameSource','Input port',...
                         'OutputValue','Horizontal and vertical components in complex form');

ax(1) = subplot(1,3,1); im(1) = imagesc(gray1,[0,1]); 
ax(2) = subplot(1,3,2); im(2) = imagesc(hsvimg,[0,1]);
ax(3) = subplot(1,3,3); 
                     
for i = 2:nFrame
 gray2 = normFrame(data(:,:,i),16,clim);
 
 uv = step(hof,gray2,gray1); %estimate optical flow for u,v
 u = real(uv);
 v = imag(uv); %uv must be complex!

 [ang,mag] = cart2pol(u,v);
 mag = mag / max(max(mag)); %normalize
 mag(mag>0.1) = 1; %clip 
 ang = mod(ang * 180/pi,360)/360; %normalize angle -- may not be quite right
 hsvimg(:,:,1) = ang;
 hsvimg(:,:,3) = 1; %mag <-- was having issue with dynamic range, but setting to 1 gets the job done
 rgbimg = hsv2rgb(hsvimg);
 try
 %can be done with subplots or whatever
 set(im(1),'cdata',gray2)
 set(im(2),'cdata',rgbimg) %HSV representation of optical flow between gray2,gray1  <-- color represents direction!
 subplot(1,3,3), quiver(u(1:20:end,1:20:end),v(1:20:end,1:20:end))  % quiver is downsampled by 20 to allow one to see not too dense arrows
 catch
     pause
 end
 
 gray1 = gray2;
 drawnow
end %for
end