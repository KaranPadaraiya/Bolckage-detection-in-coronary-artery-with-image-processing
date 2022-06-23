clear all 
close all
A=imread('uniform.png');
figure,imshow(A)

A=ordfilt2(A,5,ones(3,3));
%i=rgb2gray(j);
AInv = imcomplement(A);

BInv = imreducehaze(AInv);
B = imcomplement(BInv);
BInv = imreducehaze(AInv,0.9, 'Method','approx','ContrastEnhancement','boost');

BImp = imcomplement(BInv);
figure, montage({A BImp});
A=BImp
%===============Denoising==============================

% figure,image(wcodemat(BImp,100))
% title('denoiced Image')
%============adaptive threshold========================


T = adaptthresh(A,0.6,'NeighborhoodSize',11);
figure
imshow(T)

BW = imbinarize(A,T);

SE=strel('square',3);
%BW=imopen(BW,SE);

figure
imshow(BW)

%======================================================
A=ordfilt2(Bimp,5,ones(3,3));
%i=rgb2gray(j);
AInv = imcomplement(A);

BInv = imreducehaze(AInv);
B = imcomplement(BInv);
BInv = imreducehaze(AInv, 'Method','approx','ContrastEnhancement','boost');
BImp = imcomplement(BInv);
figure, montage({A BImp});


Bimp=adapthisteq(BImp)
figure, montage({Bimp BImp});



%B = imguidedfilter(BImp);
i=gray2d(A,BImp);

function [outImg]=gray2d(Img,bw)
[r, c]=size(Img);
bw=imresize(bw,[r,c]);
outImg=Img
 for x=1:3;
    for i=1:r
       for j=1:c
           outImg(i,j,x)=bw(i,j);
       end
    end
 end
end




%{
====================Hessian and frangi filter=========================
eig2image(Dxx,Dxy,Dyy)
eig3volume.m
FrangiFilter2D(I, options)
FrangiFilter3D(I,options)
Hessian2D(I,Sigma)
Hessian3D(Volume,Sigma)
imgaussian(I,sigma,siz)

function [Lambda1,Lambda2,Ix,Iy]=eig2image(Dxx,Dxy,Dyy)
% This function eig2image calculates the eigen values from the
% hessian matrix, sorted by abs value. And gives the direction
% of the ridge (eigenvector smallest eigenvalue) .
% 
% [Lambda1,Lambda2,Ix,Iy]=eig2image(Dxx,Dxy,Dyy)
%
%
% | Dxx  Dxy |
% |          |
% | Dxy  Dyy |
% Compute the eigenvectors of J, v1 and v2
tmp = sqrt((Dxx - Dyy).^2 + 4*Dxy.^2);
v2x = 2*Dxy; v2y = Dyy - Dxx + tmp;
% Normalize
mag = sqrt(v2x.^2 + v2y.^2); i = (mag ~= 0);
v2x(i) = v2x(i)./mag(i);
v2y(i) = v2y(i)./mag(i);
% The eigenvectors are orthogonal
v1x = -v2y; 
v1y = v2x;
% Compute the eigenvalues
mu1 = 0.5*(Dxx + Dyy + tmp);
mu2 = 0.5*(Dxx + Dyy - tmp);
% Sort eigen values by absolute value abs(Lambda1)<abs(Lambda2)
check=abs(mu1)>abs(mu2);
Lambda1=mu1; Lambda1(check)=mu2(check);
Lambda2=mu2; Lambda2(check)=mu1(check);
Ix=v1x; Ix(check)=v2x(check);
Iy=v1y; Iy(check)=v2y(check);
end
function [Dxx,Dxy,Dyy] = Hessian2D(I,Sigma)
%  This function Hessian2 Filters the image with 2nd derivatives of a 
%  Gaussian with parameter Sigma.
% 
% [Dxx,Dxy,Dyy] = Hessian2(I,Sigma);
% 
% inputs,
%   I : The image, class preferable double or single
%   Sigma : The sigma of the gaussian kernel used
%
% outputs,
%   Dxx, Dxy, Dyy: The 2nd derivatives
%
% example,
%   I = im2double(imread('moon.tif'));
%   [Dxx,Dxy,Dyy] = Hessian2(I,2);
%   figure, imshow(Dxx,[]);
%
% Function is written by D.Kroon University of Twente (June 2009)
if nargin < 2, Sigma = 1; end
% Make kernel coordinates
[X,Y]   = ndgrid(-round(3*Sigma):round(3*Sigma));
% Build the gaussian 2nd derivatives filters
DGaussxx = 1/(2*pi*Sigma^4) * (X.^2/Sigma^2 - 1) .* exp(-(X.^2 + Y.^2)/(2*Sigma^2));
DGaussxy = 1/(2*pi*Sigma^6) * (X .* Y)           .* exp(-(X.^2 + Y.^2)/(2*Sigma^2));
DGaussyy = DGaussxx';
Dxx = imfilter(I,DGaussxx,'conv');
Dxy = imfilter(I,DGaussxy,'conv');
Dyy = imfilter(I,DGaussyy,'conv');
end


I=double(B);
In=I;
mask1=[1, 0, -1;1, 0, -1;1, 0, -1];
mask2=[1, 1, 1;0, 0, 0;-1, -1, -1];
mask3=[0, -1, -1;1, 0, -1;1, 1, 0];
mask4=[1, 1, 0;1, 0, -1;0, -1, -1];
  
mask1=flipud(mask1);
mask1=fliplr(mask1);
mask2=flipud(mask2);
mask2=fliplr(mask2);
mask3=flipud(mask3);
mask3=fliplr(mask3);
mask4=flipud(mask4);
mask4=fliplr(mask4);
  
for i=2:size(I, 1)-1
    for j=2:size(I, 2)-1
        neighbour_matrix1=mask1.*In(i-1:i+1, j-1:j+1);
        avg_value1=sum(neighbour_matrix1(:));
  
        neighbour_matrix2=mask2.*In(i-1:i+1, j-1:j+1);
        avg_value2=sum(neighbour_matrix2(:));
  
        neighbour_matrix3=mask3.*In(i-1:i+1, j-1:j+1);
        avg_value3=sum(neighbour_matrix3(:));
  
        neighbour_matrix4=mask4.*In(i-1:i+1, j-1:j+1);
        avg_value4=sum(neighbour_matrix4(:));
  
        %using max function for detection of final edges
        I(i, j)=max([avg_value1, avg_value2, avg_value3, avg_value4]);
  
    end
end
figure, imshow(uint8(I));

alpha=0.4;
omega=0.4;
c=(1-2.*exp(-alpha).*cos(omega)+exp(-2.*alpha))/(exp(-alpha).*(sin(omega)));
k=((1-2.*exp(-alpha).*cos(omega)+exp(-2.*alpha))*(alpha.*alpha+omega.*omega))/(2.*alpha*exp(-alpha).*sin(omega)+omega-omega.*exp(-2.*alpha));
[m,n]=meshgrid(-20:1:20,-20:1:20);
Xc=(-c).*exp(-alpha.*abs(m)).*(sin(omega)).*(m);
Xt=(k).*(alpha.*sin(omega).*abs(n)+(omega).*cos(omega).*abs(n))*(exp(-alpha.*abs(n)));
Yc=(-c).*exp(-alpha.*abs(n)).*(sin(omega)).*(n);
Yt=(k).*(alpha.*sin(omega).*abs(m)+(omega).*cos(omega).*abs(m))*(exp(-alpha.*abs(m)));
X=Xc.*Xt;
Y=Yc.*Yt;
X=X./(alpha.*alpha+omega.*omega);
Y=Y./(alpha.*alpha+omega.*omega);

imgg = derichefilter1(X, Y, Xc, Yc, Xt);
imgg=imfilter(B,imgg);
figure,imshow(imgg);
function imgg = derichefilter1(x, k, a, b, c)
osize = size(x);
x = double(x);
a = double(a);
b = double(b);
c = double(c);
k = double(k);
y1 = zeros(osize(1),osize(2));
y2 = zeros(osize(1),osize(2));
y1(:,1) = a(1)*x(:,1);
y1(:,2) = a(1)*x(:,2) + a(2)*x(:,1) + b(1)*y1(:,1);
for ii=3:osize(2)
    y1(:,ii) = a(1)*x(:,ii) + a(2)*x(:,ii-1) + b(1)*y1(:,ii-1) + b(2)*y1(:,ii-2);
end

y2(:,osize(2)-1) = a(3)*x(osize(2));
for ii=(osize(2)-2):-1:1
    y2(:,ii) = a(3)*x(:,ii+1) + a(4)*x(:,ii+2) + b(1)*y2(:,ii+1) + b(2)*y2(:,ii+2);
end
imgg = c*(y1+y2);

function imgg = derichefilter2(x, k, a, b, c)
imgg = derichefilter1(x,k,a(1:4),b,c(1));
imgg = (derichefilter1(imgg',k,a(5:8),b,c(2)))';

function [mask magn] = loc(x, y)
magn = sqrt(x.^2 + y.^2);
argu = atan2(y,x);
argu = argu/pi*4;
argu = int32(round(argu));
argu(argu == 4) = 0;
argu(argu < 0) = argu(argu < 0) + 4;
mask = boolean(zeros(size(x)));
for ii = 2:(size(x,1)-1)
    for jj = 2:(size(x,2)-1)
        switch argu(ii,jj)
            case 0
                mask(ii,jj) = (max(magn(ii,jj+1),magn(ii,jj-1)) <= magn(ii,jj));
            case 1
                mask(ii,jj) = (max(magn(ii-1,jj+1),magn(ii+1,jj-1)) <= magn(ii,jj));
            case 2
                mask(ii,jj) = (max(magn(ii-1,jj),magn(ii+1,jj)) <= magn(ii,jj));
            case 3
                mask(ii,jj) = (max(magn(ii+1,jj+1),magn(ii-1,jj-1)) <= magn(ii,jj));
        end
    end
end

function imgg = hystthres(x,Tl,Th)
imgg = (x>Th);
limg = (x>=Tl);
osize = size(x);
nTh = 0;
for ii = 1:osize(1)
    for jj = 1:osize(2)
        if imgg(ii,jj)
            nTh = nTh + 1;
        end
    end
end
c = zeros(1,nTh); r = zeros(1,nTh); nTh=0;
for ii = 1:osize(1)
    for jj = 1:osize(2)
        if imgg(ii,jj)
            nTh = nTh + 1;
            c(nTh) = ii; r(nTh) = jj;
        end
    end
end
imgg = bwselect(limg,r,c,8);

function imgg = derichecomplete(x, alph, Tl, Th)
k = (1 - exp(-alph))^2/(1 + 2*alph*exp(-alph) - exp(-2*alph));
as = zeros(1,8);
as(1) = k;
as(2) = k*exp(-alph)*(alph-1);
as(3) = k*exp(-alph)*(alph+1);
as(4) = -k*exp(-2*alph);
as(5:8)=as(1:4);
b = zeros(1,2);
b(1) = 2*exp(-alph);
b(2) = -exp(-2*alph);
cs = [1,1];
ax = [0,1,-1,0,as(5:8)];
cx = [-(1 - exp(-alph))^2,1];
ay = [ax(5:8),ax(1:4)];
cy = [cx(2) cx(1)];

deriches = derichefilter2(x, k, as, b, cs);
derichex = derichefilter2(deriches, k, ax, b, cx);
derichey = derichefilter2(deriches, k, ay, b, cy);
[mask mag] = nonmaxsupp(derichex, derichey);
mag(~mask) = 0;
imgg = hystthres(mag,Tl,Th);

clc; clear all; close all;
imagepath = input('Enter the image path in single quotes: ');
alph = input('Enter the value of alpha to be used: ');
Tl = input('Enter the value of Tl to be used: ');
Th = input('Enter the value of Th to be used: ');
imgg = imread(imagepath);
szzz = size(size(imgg));
if szzz(2) == 3
    osize = size(imgg);
    hystf = boolean(zeros(osize(1:2)));
    for ii=1:3
        hystf = hystf | derichecomplete(imgg(:,:,ii),alph,Tl,Th);
    end
else    
    hystf = derichecomplete(imgg,alph,Tl,Th);
end
end
end
end
end
end

=====================================================================
%}


%{
eg=edge(B,"deriche");
x=bwareaopen(eg,10);
figure, montage({eg,x});
SE=strel("line",5,45);
op=imopen(eg,SE);

eg=edge(B,"canny",0.1,3);
figure,imshow(eg);
title("vein")
figure;
imcontour(B)

H = fspecial('average',100);
B=imfilter(B,H);
B=B*1.5;
B=imcomplement(B);
A=C-B +30;
B=imadjust(A);
imshow([B,A]);



%}
