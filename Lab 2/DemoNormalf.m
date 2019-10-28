function [slumptal] = DemoNormalf(N, medel, varians)
% Anrop: s = DemoNormalf(N, m, v);
% Funktionen skapar N st slumptal ur normalfördelning, med väntevärde 
% (medelvärde) m och varians v. Om m och/eller v utelämnas används 
% defultvärdena m = 0, v = 1.
% m alternativt v kan vara vektorer - funktionen beräknar slumptal
% för varje värde på m alternativt v och ritar flera figurer. Både v och m
% kan inte ges som vektorer samtidigt. 
% Funktionen ritar dels upp slumptalen och skapar ett histogram med 
% 100 staplar, där staplarnas höjd anger antalet slumptal i respektive
% stapels intervall.
% Exempel:
% DemoNormalf(500); => 500 slumptal, väntevärde 0, varians 1
% s = DemoNormalf(100, 5, 3); => 100 slumptal, väntevärde 5, varians 3. 
%                                Slumptalen lagras i s
% DemoNormalf(100, [0 10], 5); => 100 slumptal med väntevärde 0 respektive
%                                 10, varians 5 i båda fallen

if nargin < 3
    varians = 1;
end
if nargin < 2
    medel = 0;
end
close all;         % Stäng alla eventuella grafikfönster

% Beräkna slumptalen
if (length(medel)) == 1 & (length(varians) == 1)
   slumptal = medel + sqrt(varians)*randn(N,1); 
elseif (length(medel)) > 1 & (length(varians) == 1)
   for i = 1:length(medel)
       slumptal(:,i) = medel(i) + sqrt(varians)*randn(N,1);
   end
   varians = repmat(varians,1,length(medel));
elseif (length(medel)) == 1 & (length(varians) > 1)
   for i = 1:length(varians)
       slumptal(:,i) = medel + sqrt(varians(i))*randn(N,1);    
   end
   medel = repmat(medel,1,length(varians));
else
    error('Ej tillåtet att ge flera varianser och flera väntevärden samtidigt.');
end

% Utskrift
for i=1:max([length(medel) length(varians)])
    titelstr = [num2str(N), ' slumptal ur normalfördelning. Medelvärde = ',...
    num2str(medel(i)), ' varians = ' num2str(varians(i))];
    figure('Name',titelstr,'NumberTitle','off');
    subplot(2,1,1);
    plot(slumptal(:,i),'ro');
    subplot(2,1,2);
    title('Histogram över antal element i respektive stapel');
    hist(slumptal(:,i),100);
end