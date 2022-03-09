clc; clear;

Error=-2:0.2:2;

L1=abs(Error);
L2=Error.^2;

sigma=0.25;
lambda=1/(2*(sigma)^2);
G0=1/(sqrt(2*pi)*sigma);
G1=1/(sqrt(2*pi)*sigma).*exp(-lambda*(Error).^2);
Correntropy_025=G0-G1;

sigma=0.5;
lambda=1/(2*(sigma)^2);
G0=1/(sqrt(2*pi)*sigma);
G1=1/(sqrt(2*pi)*sigma).*exp(-lambda*(Error).^2);
Correntropy_05=G0-G1;

sigma=0.75;
lambda=1/(2*(sigma)^2);
G0=1/(sqrt(2*pi)*sigma);
G1=1/(sqrt(2*pi)*sigma).*exp(-lambda*(Error).^2);
Correntropy_075=G0-G1;

figure(1)
plot(Error,L1)
hold on
plot(Error,L2)
plot(Error,Correntropy_025)
plot(Error,Correntropy_05)
plot(Error,Correntropy_075)
hold off
title('Loss Function Comparision')
xlabel('Error')
ylabel('Loss')
legend('L1 Loss','L2 Loss','C-Loss \sigma=0.25','C-Loss \sigma=0.5','C-Loss \sigma=0.75')
