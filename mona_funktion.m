%Kapitel 1: Numeriska fel och finita differenser.
%F�r att l�sa finita differenser kr�vs tre inputs: x-v�rde, stegl�ngd h och
%funktionen f.
%Exempel p� uppgifter: Approximera derivatan f�r funktionen f(x).

%Fram�tdifferens
%{
framaat = @(x,h) (f(x+h)-f(x))/h;
f_h = framaat(x,h);
f_h_halva = framaat(x,h/2);
%}
%Centraldifferens
%{
central = @(x,h) (f(x+h)-f(x-h))/(2*h);
c_h = central(x,h)
c_h_halva = central(x,h/2)
%}
%Bak�tdifferens
%{
bakaat = @(x,h) (f(x)-f(x-h))/h
b_h = bakaat(f,h)
b_h_halva = bakaat(x,h/2)
%}
%Richardson extrapolation
%{
F�r fram�tdifferens:
rec_b = 2*f_h_halva-f_h
alternativt,
Richexframaat = f_h_halva + (f_h_halva - f_h) / (2^(k)-1);
F�r centraldifferens:
(4*c_h_halva-c_h)/3
alternativt,
RichexCent = c_h_halva + (c_h_halva - c_h) / (4^(k)-1);
F�r bak�tdifferens:
k = 1;
RichexBak = b_h_halva + (b_h_halva - b_h) / (2^(k)-1);


%}


%Kapitel 2: Sekantmetoden och Newton-Rhapsons metod.
%Anv�nds f�r ekvationsl�sning (ex. hitta alla l�sningar f�r f(x) = 0).
%Intervallhalveringsmetoden (Kr�vs f�ljande inputs: funktion, tolerans e,
%intervall a till b).
%{
function [a] = Intervallhalvering(a,b,funktioneniuppgiften,e)
while funktioneniuppgiften(a)*funktioneniuppgiften(b) < 0 && (b - a) > e
    c = 1/2*(a+b);
    if funktioneniuppgiften(a)*funktioneniuppgiften(c) < 0
        b = c;
    else
        a = c;
    end
end
end
%}
%Newton-Rhapson metod (Kr�vs f�ljande inputs: funktion, derivatan av
%funktionen, startv�rde x, tolerans e, max iterationer N).
%{
function [x,j] = newtonrhapson(funktioneniuppgiften,funktioneniuppgiften_prim,x,e,N)
d = funktioneniuppgiften(x)/funktioneniuppgiften_prim(x);
x = x - d;
j = 1;
while abs(d) >= e && j <= N
    d = funktioneniuppgiften(x)/funktioneniuppgiften_prim(x);
    x = x - d;
    j = j + 1;
end
end
%}
%Sekantmetoden
%{
fun = @(x)exp(x)-x^2;
x1 = -10;
x2 = -0;
f1 = fun(x1);
dx = 1;
iterSek = 0;

while abs(dx) > 1e-10 && iterSek < 10
  f2 = fun(x2);
  dx = - f2*(x2 - x1)/(f2 - f1);
  x1 = x2;
  f1 = f2;
  x2 = x2 + dx;
  iterSek = iterSek + 1;
end
disp(x1);
iterSek
%}
%Regula-falsi-metoden
%{
format long
clc;
close all;
clear all;
syms x;
f=exp(x)-x^2; %Enter the Function here
n=input('Enter the number of decimal places:');
epsilon = 10^-(n+1)
x0 = input('Enter the 1st approximation:');
x1 = input('Enter the 2nd approximation:');
for i=1:100
    f0=vpa(subs(f,x,x0)); %Calculating the value of function at x0
    f1=vpa(subs(f,x,x1)); %Calculating the value of function at x1
y=x1-((x1-x0)/(f1-f0))*f1; %[x0,x1] is the interval of the root
err=abs(y-x1);
if err<epsilon %checking the amount of error at each iteration
break
end
f2=vpa(subs(f,x,y));
if (f1)*(f2)<1
x0=y;  %taking the next interval as[x0,x1] = [y,x1]
x1=x1;
else
    x0=x0; %taking the next interval as[x0,x1] = [x0,y]
    x1=y;
end
end
y = y - rem(y,10^-n); %Displaying upto required decimal places
fprintf('The Root is : %f \n',y);
%}


%Kapitel 3: Linj�r algebra, normer och iterativa metoder.
%Jacobis metod
%{
%Jacobi Method
%Solution of x in Ax=b using Jacobi Method
% * _*Initailize 'A' 'b' & intial guess 'x'*_

A=[5 -2 3 0;-3 9 1 -2;2 -1 -7 1; 4 3 -5 7]
b=[-1 2 3 0.5]'
x=[0 0 0 0]'
n=size(x,1);
normVal=Inf; 
 
% * _*Tolerence for method*_
tol=1e-5; itr=0;
%Algorithm: Jacobi Method

format long
while normVal>tol
    xold=x;
    
    for i=1:n
        sigma=0;
        
        for j=1:n
            
            if j~=i
                sigma=sigma+A(i,j)*x(j);
            end
            
        end
        
        x(i)=(1/A(i,i))*(b(i)-sigma);
    end
    
    itr=itr+1;
    normVal=abs(xold-x);
end

fprintf('Solution of the system is : \n%f\n%f\n%f\n%f in %d iterations',x,itr);
%}
%Gauss Seidel metod
%{
% Solution of x in Ax=b using Gauss Seidel Method
% * _*Initailize 'A' 'b' & intial guess 'x'*_
%
A=[5 -2 3 0;-3 9 1 -2;2 -1 -7 1; 4 3 -5 7]
b=[-1 2 3 0.5]'
x=[0 0 0 0]'
n=size(x,1);
normVal=Inf; 
% 
% * _*Tolerence for method*_
tol=1e-5; itr=0;
% Algorithm: Gauss Seidel Method
%
while normVal>tol
    x_old=x;
    
    for i=1:n
        
        sigma=0;
        
        for j=1:i-1
                sigma=sigma+A(i,j)*x(j);
        end
        
        for j=i+1:n
                sigma=sigma+A(i,j)*x_old(j);
        end
        
        x(i)=(1/A(i,i))*(b(i)-sigma);
    end
    
    itr=itr+1;
    normVal=norm(x_old-x);
end
%
fprintf('Solution of the system is : \n%f\n%f\n%f\n%f in %d iterations',x,itr);
%}
%LR-faktorisering: googla p� det. Finns en inbyggd funktion i matlab (kallas f�r LU matrix factorization).

%Kapitel 4: Numerisk integration.
%(se integration.m)

%Kapitel 5: Monte-carlo metoder.
%Monte-carlo l�sning p� en integral.
%{
function [val,err] = mcint(f,D,N,M) 
% MCINT.  Function that solves an integral using Monte Carlo.
%
% Calling sequence: [val,err] = mcint(@fun,D,N,M)    
% Input:  fun - function defining the integrand. 
%               fun will be evaluated in dim*N points, where dim is the
%               dimension of the problem. 
%         D   - the domain. Each row in the matrix D represents the range 
%               of the corresponding variable. For example, a 3-dimensional
%               integral solved on the unit cube would be given by             
%               D = [0 1; 0 1; 0 1];
%         N   - the number of points in each realization.
%         M   - the number of repetitions used for error estimation. 
%               (Recommendation, M = 30+).
%               Total number of points used is thus M*N.  
%
% Output:  val - the resulting integral value
%          err - the error in the result (the standard deviation)

    V   = cumprod(D(:,2)-D(:,1));
    dim = numel(V);
    V   = V(end);
    
    r=zeros(dim,N);
    
    for j=1:M
        r(:,:) = repmat(D(:,1),1,N)+rand(dim,N).*repmat(D(:,2)-D(:,1),1,N);
        I(j)=V*mean(f(r));
    end
    
    val = mean(I);
    err = std(I);
    
end
%}

%Kapitel 6: Minsta-kvadrat-metoden.
%(Se projektet.)

%Kapitel 7: Markovkedjor. Titta p� Sellgrens skit.
















%Sellgrens l�sningar:
%Fram�tdifferenser
%{
format long
f = @(x,h)(atan(1+x^2)); %Funktion
x0 = 1/2; %Derivatan n�ra v�rde x = 1/2

h1=0.02; %Stegl�ngd
h2=0.01; %Stegl�ngd

df1 = (f(x0+h1)-f(x0))/(h1) %fram�tdifferens med stegl�ngd h1 = 0.02 utan extrapolation
df2 = (f(x0+h2)-f(x0))/(h2) %fram�tdifferens med stegl�ngd h2 = 0.01 utan extrapolation

REC = ((2*df2)-df1) % Richardsson extrapolation

f_prime = ((2*x0)/((x0^2 + 1)^2 + 1)); %exakta v�rdet vid x0

edc1 = abs(f_prime-df1); %Error hur l�ngt ifr�n man var riktiga v�rdet med stegl�ngd h1
edc2 = abs(f_prime-df2); %Error hur l�ngt ifr�n man var riktiga v�rdet med stegl�ngd h2
eREC = abs(f_prime-REC); %Error hur l�ngt ifr�n man var riktiga v�rdet med Richardsson extrapolation
Resultat = [df1 df2 REC]
Error = [edc1 edc2 eREC]
%}
%Centraldifferenser
%{
format long
f = @(x,h)(atan(1+x^2)); %Funktion
x0 = 1/2; %Derivatan n�ra v�rde x = 1/2

h1=0.02; %Stegl�ngd
h2=0.01; %Stegl�ngd

dc1 = (f(x0+h1)-f(x0-h1))/(2*h1) %centraldifferens med stegl�ngd h1 = 0.02 utan extrapolation
dc2 = (f(x0+h2)-f(x0-h2))/(2*h2) %centraldifferens med stegl�ngd h2 = 0.01 utan extrapolation

REC = (dc2+(dc2-dc1)/3) % Richardsson extrapolation

f_prime = ((2*x0)/((x0^2 + 1)^2 + 1)); %exakta v�rdet vid x0

edc1 = abs(f_prime-dc1); %Error hur l�ngt ifr�n man var riktiga v�rdet med stegl�ngd h1
edc2 = abs(f_prime-dc2); %Error hur l�ngt ifr�n man var riktiga v�rdet med stegl�ngd h2
eREC = abs(f_prime-REC); %Error hur l�ngt ifr�n man var riktiga v�rdet med Richardsson extrapolation
Resultat = [dc1 dc2 REC]
Error = [edc1 edc2 eREC]
%}
%L�s med valfri metod i kursen (utom kurvritning).
%{
% a)
%a) En familj skall anl�gga en simbass�ng f�r barnen med vattendjupet x m,
%bredden (5x+1) m och l�ngden (15x+1) m.
%Bass�ngen har formen av ett r�tblock.
%Ber�kna vattendjupet om bass�ngen skall rymma 400 m3 vatten.

% definiera funktionen
f=@(x) 75*x.^3+20*x.^2+x-400;
% definiera funktionens derivata
df=@(x) 225*x.^2+40.*x+1;

% plotta
figure;
x=0:0.01:3;
plot(x,f(x))
grid
% vi ser att rimligt startv�rde kan vara x=1
% k�r Newton-Raphson
x=1; % start punkt
p=f(x);
n_iter=0;
while abs(p)>0.001
    x=x-f(x)/df(x);
    p=f(x);
    n_iter=n_iter+1;
end

% skriv ut svaret
x
%}
%Ber�kna roten ur 3 med valfri metod i kursen.
%{


% Newton-Raphson igen:
f=@(x) x^2-3;
df=@(x) 2*x;
% Newton-Raphson:
x=2; % starting point
p=f(x);
n_iter=0;
while abs(p)>0.001
    x=x-f(x)/df(x);
    p=f(x);
    n_iter=n_iter+1;
end
x
sqrt(3)
%}
%Centralformeln
%{
x=0.5;
h=0.2;
format long

% a)
% centralformeln:
d1=(atan(x+h)-atan(x-h))/(2*h)

% b)
% halvera h
d2=(atan(x+h/2)-atan(x-h/2))/(2*h/2)
% felet i centralformeln �r O(h^2)
% vi har alltsa d~=d1+k*h^2 och d~=d2+k*(h/2)^2,
% d�r d �r exakt v�rde. L�s m.a.p. k:
k=(d1-d2)/(h^2/4-h^2);
% extrapolerat v?rde:
fp_extra = d1+k*h^2
% exakt:
fp_exact=1/(1+x^2)
%}
%Exakta svaret p� fPrime(1/2)
%{
format rat
f_prime = ((2*(1/2))/(((1/2)^2 + 1)^2 + 1)) %exakta v�rdet vid x0 = 16/41

%}
%Intervallhalveringsmetoden f�r att hitta f(x)=0 i [2,3]
%{
format long

a = 2; %start p� intervall
b = 3; %slutet p� intervall

if funktionIUppgiften(a) * funktionIUppgiften(b) < 0
    while b-a > 0.01 % loopar s�l�nge intervallets l�ngd �r st�rre �n 0.01
        c = (b-a) / 2; %halva interfallet
        if funktionIUppgiften(b-c)*funktionIUppgiften(b) < 0
            a = a + c;
        elseif funktionIUppgiften(a) * funktionIUppgiften(a+c) <= 0
            b = b-c;
        end
    end
end
x = a + c;
funktionIUppgiften(x)
%Funktion f�r att kolla funktionsv�rdet
function y = funktionIUppgiften( x )
%Returns the y-value of the inputted x-value
y = exp(x) - 2*x^2;
end
%}
%Newton Rhapson f�r att hitta f(x)=0 i [1,3]
%{
function [ x,n ] = NewRap( x0 )
    n = 1;
    funktionIUppgiften = @(x) x + cos(x);
    funktionIUppgiften_prim = @(x) 1 - sin(x);
    x = x0 - funktionIUppgiften(x0) / funktionIUppgiften_prim(x0); %Newtonraphson
    while abs(x - x0) >= 10^-7 %stannar n�r differancen �r mindre 10^-7
    x0 = x;
    x = x0 - funktionIUppgiften(x0) / funktionIUppgiften_prim(x0); %Assign the new x value using Newton Rhapsons method
    n = n + 1;
    end
    disp(n);
    disp(x);
end

%Funktion f�r att kolla funktionsv�rdet
function y = funktionIUppgiften( x )
%Returns the y-value of the inputted x-value
y = exp(x) - 2*x^2;
end

function y = funktionIUppgiften_prim( x )
%Returns the y-value of the inputted x-value
y = exp(x) - 4*x;

end
%}
%Monte Carlo f�r att hitta l�sning till integral med gr�nser
%{
fx = @(x)log(x);
Resultat = mcint(fx,[1 exp(1)], 10, 1000)

function [val,err] = mcint(f,D,N,M)
%MCINT. Function that solves an integral using MonteCarlo
    V = cumprod(D(:,2)-D(:,1));
    dim = numel(V);
    V = V(end);
    
    r= zeros(dim,N);
    
    for j = 1:M
        r(:,:) = repmat(D(:,1),1,N) + rand(dim,N).*repmat(D(:,2)-D(:,1),1,N);
        I(j) = V*mean(f(r));
    end
    
    val = mean(I);
    err = std(I);

end
%}
%Trapetsregeln med stegl�ng h functionen f.
%{
resultat = trapets(@log,1,exp(1),10)
function M = trapets(f,a,b,n)
%h =(b-a)/n;
h = (exp(1)-1)/10;
x = a:h:b;
y = f(x);
M = h*(sum(y)-y(1)/2-y(end)/2);
end
%}
% L�t Matlab L�sa integralen
%{
func = @(x) log(x)
value=integral(func,1,exp(1))

%}
%Best�m samtliga l�sningar till ekavationen f
%{
f = @(x)(x.^2-5.*x-5-cos(x)); %funktionen f
fp = @(x)(2.*x-5+sin(x)); %derivatan av f

x = -2; %startgissning f�r x

tol = 1e-7; %noggrannheten i l�sningen

n = 0; %antal iterationer som initialt ?r 0

while abs(f(x)/fp(x))>tol
     
      x = x-f(x)/fp(x); %nytt x-v�rde
      n = n+1; %�kar antal iterationer
      
end

x %l�sningen till ekvationen, d.v.s. f(x) ~ 0
n %antal iterationer 


%}
%Hitta maximum till funktionen med Monte Carlo KOLLA P� DENNA
%{
x = -3:0.1:3;
y = -3:0.1:3;
f = zeros(length(y),length(x));

for i=1:length(x)
    for j=1:length(y)
        f(i,j) = -(x(i).^2+y(j).^2+2*x(i)+3*y(j)+(sin(x(i)+y(j))).^2 + 2);
    end
end

figure; 
surf(x,y,f); 
axis tight

%F�rs�ker hitta minimum:
xy_max = 3; %x,y-max
xy_min = -3; %x,y-min

N = 100; %antal slumptal

f_max = -1/eps; %startv�rde f�r f_max (litet)
x_max = 0; %startv�rde f�r x_max
y_max = 0; %startv�rde f�r y_max

for i=1:N
    x = xy_min+(xy_max-xy_min)*rand(1,1);
    y = xy_min+(xy_max-xy_min)*rand(1,1);
    f = -(x^2+y^2+2*x+3*y+(sin(x+y))^2 + 2);
    if f>f_max
        x_max = x;
        y_max = y;
        f_max = f;
    end
end
hold on;

plot3(x_max,y_max,f_max,'r*','markersize',15)
hold off

f_max
x_max
y_max
%}
%{
function [coef] = PolyReg(x,y,order)

    matrix = zeros(order+1,order+1);
    right = zeros(order+1,1);
    powers = zeros(2*order+1,1);
    
    for i = 0:(2*order)
        powers(i+1) = sum(x.^i); 
    end
    
    for i = 0:order
        right(i+1) = sum(y.*(x.^i)); 
    end
    
    for i = 0:order
        matrix(i+1,:) = powers((i+1):(i+1+order)); 
    end
    
    coef = flip(matrix\right);

end

%}
%Ber�kna omr�dets volym med Monte Carlo


N = 1000:1000:50000; %olika antal slumptal

for n=1:length(N)
    
    N_in = 0; %antal punkter innanf�r omr�det (initialt)
    
    for i=1:N(n)
        x = rand(1,1); %slumpat v�rde f�r x
        y = rand(1,1); %slumpat v�rde f�r y
        z = rand(1,1); %slumpat v�rde f�r z
        
        if (x^2+sin(y))<=z && (x+exp(y)-z)<=1 %unders�ker om punkten (x,y,z) �r innanf�r
            N_in = N_in+1; %�kar antal v�rden innanf�r
        end
    end
    V_bes(n) = N_in/N(n); %alla v�rden p� volymerna i en array    
    
end

figure; 

plot(N, V_bes,'.-'); 

xlabel('N')
ylabel('V?rden f?r volymerna')


%Vi s�ger att sista v�rdet f�r V_bes ?r en bra approximation att anv�nda d�
%vi ska kolla p� felet:

err_V = abs(V_bes(1:end-1)-V_bes(end)); %absoluta felet


plot(N(1:end-1), err_V,'.-'); 

xlabel('N')
ylabel('error V�rden f�r volymerna')

plot(log(N(1:end-1)), log(err_V),'.-'); 

xlabel('N')
ylabel('V�rden f�r volymerna')

%Hur felet beror av N kan man se fr�n en r�t linje
%

%Uppgift 5 181003


P = [0.06 0.76 0.06 0.06 0.06;
    0.3 0 0.7 0 0;
    0 0.3 0 0.7 0;
    0 0 0.3 0 0.7;
    0 0 0 0.3 0.7]; %Array med sannolikheter
s = [1.0 0.0 0.0 0.0 0.0];
n = 8;
test = true;
if s == s*P
    test = false;
end
while test
    if s == s*P
        test = false;
    end
    s = s*P;
    n = n+1;
end
M = sum ([1 2 3 4 5] .* s);
mean(M)% Viktat medelv�rde - medelantal kunder