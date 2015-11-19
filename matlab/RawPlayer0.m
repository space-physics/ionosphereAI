function data =RawPlayer0(dataFN,varargin)
%% plays back raw FM audio
% NOTE: you must have first used the Echotek Python binary .bin to .mat converter, sorry
%
% some examples
% RawPlayer0('~/data/2010-08-03/rx40/isis_rf_89.70_fm@1280877299.mat')
%--------------
% RawPlayer0('~/data/2010-08-03/rx51/isis_rf_103.50_fm@1280877299.mat'); %brief strong return at 10 sec.
% RawPlayer0('~/data/2010-08-03/rx40/isis_rf_103.50_fm@1280877299.mat']; 
%------------------
% RawPlayer0('~/data/2013-11-20/mat/isis_rf_89.50_fm@1384979400.mat']; % NPR talk
% RawPlayer0('~/data/2013-11-20/mat/isis_rf_106.70_fm@1384986598.mat']; % soft rock
% RawPlayer0('~/data/2011-07-27/ecdr_chip01_chan00@1311807598.bin']; % just static?
% RawPlayer0('~/data/2013-11-20/ecdr_chip01_chan00@1384979400.bin']; % just static?

p = inputParser;
addParamValue(p,'doLPF',false)
addParamValue(p,'doPlayAudio',true)
addParamValue(p,'doResamplePB',false) %#ok<*NVREPL>
addParamValue(p,'diffMethod','forward') %forward, central, angle
addParamValue(p,'sampling_frequency',150e3) %[Hz]  or 200e3 some files
parse(p,varargin{:})
U = p.Results;

%MaxDev = 75e3; %[Hz]
%km = MaxDev/15e3; %approximate FM broadcast modulation index.
%% load data
[~,~,dataExt] = fileparts(dataFN);
switch dataExt
    case '.mat'
        load(dataFN)
    case '.bin' % this is not correct, use Python code instead!
        maxSampToRead = 10e3; %arbitrary
        fid = fopen(dataFN,'r','l');
        if fid<1, error(['did not find ',dataFN]), end
        
        %pre-allocation
        ind = 0;
        dtmp(maxSampToRead,2) = nan; 
        
        while ~feof(fid) && ind<=maxSampToRead
            ind = ind+1;
            dtmp(ind,2) = fread(fid,1,'int16=>double',0,'l'); % Q 
            dtmp(ind,1) = fread(fid,1,'int16=>double',0,'l'); % I
            %dtmp = fread(fid,1,'int32=>int32',0,'l');
            %jtmp = dec2bin(dtmp,32);
            %stmp = [bin2dec(jtmp(17:end)), bin2dec(jtmp(1:16))]; %16-bit IQ packed as QI 32-bit (?)
            
            if ~mod(ind,10000), display(['processing sample ',int2str(ind)]), end
        end
         data = dtmp(:,1) + 1j*dtmp(:,2);
end 
display(['Using Sampling Frequency ',num2str(U.sampling_frequency),' Hz'])
Ns = length(data);
%% create FIR lowpass filter (to eliminate L-R and stereo carrier)

if U.doLPF
FIRorder = 50; %arbitrary
Fc = 75e3; %cutoff frequency [Hz]
Wn = Fc/U.sampling_frequency;
B = fir1(FIRorder,Wn); %hamming window, lowpass by default for fir1()
A = 1; %by def'n of FIR

    display(['Using ',int2str(FIRorder),'-tap LPF, with corner freq.: ',num2str(Fc/1e3),' kHz.'])
    data(:,1) = filter(B,A,data);
else 
    display('No LPF used.')
    data(:,1) = double(data);
end



Ts = 1/U.sampling_frequency;

t(:,1) = (0:Ns-1)*Ts; %0:Ts:60-Ts;
%% define data portions
I = real(data);
Q = imag(data);
%% demodulation 
%reference: Lyons Ch. 17
switch U.diffMethod
    case 'forward' % forward difference: Two-point stencil
        dt = Ts;
        m = [0 ; (I(1:end-1) .* diff(Q)/dt - Q(1:end-1) .* diff(I)/dt) ./...
            (I(1:end-1).^2 + Q(1:end-1).^2)];
    case 'central' % central difference: Three-point stencil
        m = ( I .* central_diff(Q,Ts) - ...
              Q .* central_diff(I,Ts) ) ./ ...
              (I.^2 + Q.^2);
    case 'angle' % just using definition of complex angle and three-point stencil
        m = central_diff( unwrap( atan2(Q,I) ), Ts);
end
%%
% normalize -- output data amplitude must be $\in$ [-1,1] or terrible distortion
maxM = max(m);    
m = m/maxM;

%clip (ugly)
% m(m>1) = 1;
% m(m<-1) = -1;

%%
so2k = U.sampling_frequency/2/1e3; %convenience (save typing)

figure(1),clf(1)
subplot(2,1,1)
plot(t,I)
ylabel('Real (I) ampl.')

subplot(2,1,2)
plot(t,Q)
ylabel('Imag (Q) ampl.')

if U.doLPF
display('plotting')
figure(2),clf(2)
freqz(B,A,512,'whole',U.sampling_frequency)
title({dataFN,'FIR filter frequency response'},'interpreter','none')

figure(3),clf(3)
impz(B,A,512,U.sampling_frequency)
title({dataFN,'FIR filter impulse response'},'interpreter','none')
end

figure(4)
[pxx,f]= pwelch(data,15000,7500,15000,U.sampling_frequency,'centered'); %arbitrary windowing
plot(f/1e3,20*log10(pxx/max(pxx)))
title({dataFN,'Post-filter baseband signal PSD'},'interpreter','none')
xlabel('freqency [kHz]')
ylabel('Amplitude [dB]')
grid on
set(gca,'ylim',[-60 0],'xlim',[-so2k so2k],...
    'xtick',-so2k:so2k/4:so2k,'ytick',-60:5:0) % an arbitary dynamic range

figure(5)
%plot(t(1:end-1),m)
plot(t,m)
xlabel('time [sec]')
ylabel('amplitude [normalized]')
title({dataFN,'Demodulated signal'},'interpreter','none')

%%
if U.doPlayAudio
display('playing audio')

if U.doResamplePB
%resample audio to a managable frequency
soundCardSR = 44100; %[Hz]
soundCardBits = 16;
[p,q] = rat(soundCardSR/U.sampling_frequency,0.0001);
mDS = resample(m,p,q);

p = audioplayer(mDS,soundCardSR,soundCardBits);
else 
p = audioplayer(m,U.sampling_frequency);
end
playblocking(p)
end

end %function
