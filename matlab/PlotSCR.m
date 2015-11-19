function PlotSCR(fn)
%%
% This program read and plot the signal to clutter ratio of passive FM
% radar by MIT Haystack Frank Lind group.
% It is perhaps the first thing one would do to manually inspect a file to
% see if something is of interest for further analysis.
%
% example
% PlotSCR('~/data/2010-08-03/rx40rx51/fm103.5/ambi/iax001_000_103.50@1280873700.h5')

[ambiguity,rangeKM,velocityMPS,t] = readFMh5(fn);

plot_SCR(ambiguity,rangeKM,velocityMPS,t,fn)


end 

function plot_SCR(ambiguity,rangeKM,velocityMPS,t,fn)
% inputs:
% t: utc second since Unix epoch

T = datestr(datetime(1970,1,1,0,0,t));

absamg = abs(ambiguity./median(ambiguity(:)));

figure
imagesc(rangeKM,velocityMPS,10*log10(absamg),[0 6])
set(gca,'ydir','normal')
colorbar
xlabel('Range [km]')
ylabel('Velocity [m/s]')

title({datestr(datetime(T)),fn,'Signal to Clutter Ratio [dB]'},'interpreter','none')

end
