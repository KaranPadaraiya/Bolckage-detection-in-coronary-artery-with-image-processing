
close all

A=imread('C:/Users/Karan Padariya/OneDrive/Desktop/Internship/Coronary Artery/Coronary Artery/1/Video frams/scene00001.png');
B=imread('C:/Users/Karan Padariya/OneDrive/Desktop/Internship/Coronary Artery/Coronary Artery/1/Video frams/scene00031.png');
A=im2gray(A);
B=im2gray(B);
C=imsubtract(B,A);
c=histeq(C);
imshow(c);










[cropped,map]=imread("croped/155image_orig.bmp");
cropped=im2gray(cropped);
figure,imshow(cropped);
crop=im2double(cropped)*255;
oned=im21d(crop);
s=std(oned);
m=mean(oned);
cv=uint8(s*100/m);




c=imcomplement(imbinarize(cropped,'global'));
c=bwareaopen(c,5);
figure,imshow(c);
% 
% out = bwmorph(c,'skel',Inf);
% black_pts=bpt2d(out);
% Phi = segCroissRegion(15,cropped_region,20,20);
cropped_region =bwmorph(c,'skel',Inf);
% cropped_region=bwmorph(cropped_region,'spur',Inf);
% figure,imshow(cropped_region);
  
 x=[];
[skel_pts,l]=bpt2d(cropped_region);
nc=64;
nr=64;
new1=zeros(nc,nr);
if l>10
   x=uint8(1:l);
   for i=1:length(x)
      Phi = segCroissRegion(cv,cropped,skel_pts(x(i),2),skel_pts(x(i),1));
      new1=im2bw(new1);
      Phi=imresize(Phi,[64 64]);
      new1=imadd(new1,Phi);
     
  end
end

figure,imshow(new1);
function [out,l]=bpt2d(bw)
[r, c]=size(bw);
x=1;
outx=[];
outy=[];
for j=1:r
    for i=1:c
        if bw(j,i)==1
            outx(x)=i;
            outy(x)=j;
            x=x+1;
%     else
        end
    end
end
out=[outx; outy];
out=transpose(out);
l=length(out);
end

function Phi = segCroissRegion(tolerance,Igray,x,y)
if(x == 0 || y == 0)
    imshow(Igray,[0 255]);
    [x,y] = ginput(1);
end
Phi = false(size(Igray,1),size(Igray,2));
ref = true(size(Igray,1),size(Igray,2));
PhiOld = Phi;
Phi(uint8(x),uint8(y)) = 1;
while(sum(Phi(:)) ~= sum(PhiOld(:)))
    PhiOld = Phi;
    segm_val = Igray(Phi);
    meanSeg = mean(segm_val);
    posVoisinsPhi = imdilate(Phi,strel('disk',1,0)) - Phi;
    voisins = find(posVoisinsPhi);
    valeursVoisins = Igray(voisins);
    Phi(voisins(valeursVoisins > meanSeg - tolerance & valeursVoisins < meanSeg + tolerance)) = 1;
end

% Uncomment this if you only want to get the region boundaries
% SE = strel('disk',1,0);
% ImErd = imerode(Phi,SE);
% Phi = Phi - ImErd;
end

% I=imread("croped/1image.bmp")
% % figure,imshow(I);
% rotI =I;
% out = bwmorph(I,'skel',Inf);
% out1 = bwmorph(out,'spur',Inf);
% 
% figure,imshow(out);
% figure,imshow(out1);
% figure,imshow(labeloverlay(Im,out,'Transparency',1))

% BW = edge(rotI,'canny');
% 
% [H,T,R] = hough(BW);
% imshow(H,[],'XData',T,'YData',R,...
%             'InitialMagnification','fit');
% xlabel('\theta'), ylabel('\rho');
% axis on, axis normal, hold on;
% 
% P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
% x = T(P(:,2)); y = R(P(:,1));
% plot(x,y,'s','color','white');
% 
% 
% 
% lines = houghlines(BW,T,R,P,'FillGap',20,'MinLength',7);
% 
% figure, imshow(rotI), hold on
% max_len = 10;
% 
% for k = 1:length(lines)
%    xy = [lines(k).point1; lines(k).point2];
%    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
% 
%    % Plot beginnings and ends of lines
%    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
% 
%    % Determine the endpoints of the longest line segment
%    len = norm(lines(k).point1 - lines(k).point2)
%    if ( len > max_len)
% %       max_len = len;
%       xy_long = xy;
%    end
% end
% 
% plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','cyan');
% 
%=======================to convert 2d image to 1d image=======================
function [outImg]=im21d(bw)
[r, c]=size(bw);

for i=1:r
    for j=1:c
        outImg((i-1)*c +j)=bw(i,j);
    end
end
end

