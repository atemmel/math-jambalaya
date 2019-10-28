function tarning_upprepa(n, N)
format long

%n = 10; %antal kast per medelv�rde
%N = [10^3 10^4 10^5]; %antal upprepningar
for i=1:length(N)
    figure;
    hold on
    r = zeros(1,N(i));
    for j=1:N(i)
        y = floor(1+6*rand(1,n));
        r(j) = mean(y);
    end
    hist(r,100); % Rita ett histogram med 100 intervall
    title([num2str(n),' st kast per medelvärde med ', num2str(N(i)), ' upprepningar'])
    xlabel('Medelvärden')
    ylabel('Frekvensen')

    hold on
    sigma = sqrt(35/(12*n));
    f = @(x)(1/(sigma*sqrt(2*pi))) * exp(-(((x - 3.5).^2))/(2*(sigma^2)));
    x = sort(r);
    y = f(x);
    plot(x, y * 1000);
end
