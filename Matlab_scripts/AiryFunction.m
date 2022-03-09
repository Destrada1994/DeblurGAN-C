function I=AiryFunction(Xdim,Ydim,Xsep,Ysep,I0,gamma,F)
% Xdim=32; %X dimension of Filter
% Ydim=32; %Y dimension of Filter
% Xsep=1.4*10^-6; %Pixel Seperation in meters
% Ysep=1.4*10^-6; %Pixel Seperation
% I0=22;  %Peak amplitude of the disk
% gamma=550*10^-9; %Wavelength of Light in meters
% F=2; %F-number of Lens

X=-((Xdim-1)*Xsep)/2:Xsep:((Xdim-1)*Xsep)/2;
Y=-((Ydim-1)*Ysep)/2:Ysep:((Ydim-1)*Ysep)/2;

I=zeros(Xdim,Ydim);

for i=1:length(X)
    for j=1:length(Y)
        r=sqrt((X(i)^2)+(Y(j)^2));
        x=(pi*r)/(gamma*F);
        if r==0
            I(i,j)=I0;
        else
            I(i,j) = I0*(2*besselj(1,x)./(x)).^2;
        end
    end
    
end

end