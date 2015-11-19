function [ambiguity,rangeKM,velocityMPS,t_utc,integration_time] = readFMh5(fn)
%%
% reads SCR from Haystack passive radar
%
% example:
% readFMh5('~/data/2010-08-03/rx40rx51/fm103.5/ambi/iax001_000_103.50@1280873700.h5')

ambiguity   = h5read(fn,'/ambiguity/ambiguity' );
ambiguity   = complex(ambiguity.r,ambiguity.i); 
rangeKM     = h5read(fn,'/ambiguity/range_axis')/1e3;
velocityMPS = h5read(fn,'/ambiguity/velocity_axis');
t_utc            = h5readatt(fn,'/ambiguity','utc_second'); 
integration_time = h5readatt(fn,'/ambiguity','integration_time');
end