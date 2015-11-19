function uint8img = graytoUint8(img,range)
%not super efficient

if nargin<2
    range(1) = min(img(:));
    range(2) = max(img(:)-range(1));
end

%{ 
%old way, caused memory / out of memory spike!
    % normalize [0,1]
    norm01=(img-range(1)) ./ ( range(2) - range(1) );

    % put in [0,255]
    norm255 = 255.*norm01;
    uint8img = uint8(norm255);
%}

uint8img = uint8(255.*(img-range(1)) ./ ( range(2) - range(1) ));

%% check for clipping
%if any(uint8img(:)==255), display('possible clipping to 255'), end

end