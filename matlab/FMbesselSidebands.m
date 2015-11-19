function FMbesselSidebands()
% Michael Hirsch
% References include:
% B. Lathi "Modern Digital and Analog Communication Systems", 4th Ed., Oxford Univ. Press, 2009.
% M. Schwartz "Information Transmission, Modulation, and Noise", 3rd Ed., McGraw-Hill 1980.

JnOrder = 2.5/3; % k_omega = 5 for broadcast FM

N = 8;
sidebands = 1:N; % consider up to Nth sideband (symmetric) (0th is NOT included see Lathi p.265)

An = besselj(sidebands,JnOrder);


figure(1),clf
plot(sidebands,20*log10(An),'.')
ylabel('Relative sideband Amplitude [dB]')
xlabel('sideband order n')
title(['sine wave modulation: Relative FM sideband amplitude for k_\omega = ',num2str(JnOrder)])
grid('on')

% linear amplitude plot
figure(2),clf
stem(sidebands,An), ylabel('Relative sideband amplitude')
xlabel('sideband order n')
title(['sine wave modulation: Relative FM sideband amplitude for k_\omega = ',num2str(JnOrder)])
grid('on')
end
