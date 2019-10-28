function [] = DemoLikformf(N, r)
% Anrop: DemoLikformf(N, r);
% Funktionen skapar och ritar upp N st slumptal ur likformig fördelning på 
% intervallet [0 r]. Om r utelämnas används intervallet [0 1]. 
% Om n är en vektor skapas flera figurer med olika antal slumptal i varje
% figur. Anropet DemoLikformf([10 10 100]) skapar t ex 3 figurer där 2
% av figurerna innehåller 10 slumptal och 1 figur inneåller 100 slumptal.
% 

if nargin < 2
    r = 1;
end
close all;         % Stäng alla eventuella grafikfönster


for i = 1:length(N)
    % Beräkna slumptalen
    slumptal = r*rand(N(i),1);
    
    % Plottning
    titelstr = [num2str(N(i)), ' slumptal ur likformig fördelning.'];
    figure('Name',titelstr,'NumberTitle','off');
    subplot(2,1,1);
    plot(slumptal,'ro');
    subplot(2,1,2);
    title('Histogram över antal element i respektive stapel');
    hist(slumptal,100);
end