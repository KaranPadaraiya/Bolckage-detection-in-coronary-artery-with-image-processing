img=imread("croped\131image_orig.bmp");
% img=rgb2gray(img);
% out=vessele_seg(img);
% function [out]=vessele_seg(img)
% img=imread("croped\image00.png");
% ig=histeq(ig);
% ig=rgb2gray(i);
[r, c, ~]=size(img);
% [~, T]=imbinarize(img,'global'); 
T=graythresh(img);
ib = imbinarize(img, T);
ib = imfill(ib,'holes');
ib=imcomplement(ib);

s1=regionprops(ib,'BoundingBox','Extrema','PixelList','ConvexArea','Area');
[size_prop,~]=size(s1);

for i=1:size_prop
    s1(i).Extrema=int8(s1(i).Extrema);  
end

% si=size(s1);
% x1=extractfield (s1,"BoundingBox")
% x2=x1;
% s=size(x1);
% j=0;
% for i=1:s(2)
%     if mod(i,4)==0
%         j=j+1;
%         x(j)=x1(i-1);
%         j=j+1;
%         x(j)=x1(i);
%       
%     end
% end

% s1(2).BoundingBox(1);
% for i= 1:size_prop
%     ig = insertShape(i,'Rectangle', s1(i).BoundingBox,'LineWidth', 1);
%    
% end

%========line array============
% x=1:c;
% y=ones(1,c);
% l1=[x;y];
% l1=transpose(l1);
% 
% x=ones(1,r)*(c+1);
% y=1:r;
% l2=[x;y];
% l2=transpose(l2);
% 
% y=ones(1,c)*(r+1);
% x=1:c;
% l3=[x;y];
% l3=transpose(l3);
% 
% x=ones(1,r);
% y=1:r;
% l4=[x;y];
% l4=transpose(l4);
% %=============================
% pos=0;
% pts_position=[];
% for i=1:size_prop(1)
%     count=0;
%     A=array2table(s1(i).Extrema,'VariableNames',{'x','y'});
%     B=array2table(l1,'VariableNames',{'x','y'});
%     C1 = intersect(A,B);
%     [x , ~]=size(C1);
%     if x~=0
%         count=count+1;
%     else
%     end  
%     A=array2table(s1(i).Extrema,'VariableNames',{'x','y'});
%     B=array2table(l2,'VariableNames',{'x','y'});
%     C2 = intersect(A,B);
%     [x , ~]=size(C2);
%     if x~=0
%         count=count+1;
%     else
%     end  
%     A=array2table(s1(i).Extrema,'VariableNames',{'x','y'});
%     B=array2table(l3,'VariableNames',{'x','y'});
%     C3 = intersect(A,B);
%     [x , ~]=size(C3);
%     if x~=0
%         count=count+1;
%     else
%     end   
%     A=array2table(s1(i).Extrema,'VariableNames',{'x','y'});
%     B=array2table(l4,'VariableNames',{'x','y'});
%     C4 = intersect(A,B);
%     [x, ~]=size(C4);
%     if x~=0
%         count=count+1;
%     else
%     end  
%     pts(i)=count;
%     if count>1
%         pos=pos+1;
%         pts_position(pos)=i;
%     end
% end
% [~,intpt]=size(pts_position);
% out=zeros(r,c);
% for k=1:intpt
%     [len,~]=size(s1(pts_position(k)).PixelList);
%     for i=1:len
%         out(s1(pts_position(k)).PixelList(i+len),s1(pts_position(k)).PixelList(i))=255;
%     end
% end
% asdf=out;
% end
out = bwmorph(ib,'skel',Inf);
out1 = bwmorph(out,'spur',Inf);

% figure,imshowpair(out,out1,"montage");
brances=out-out1;
% figure,imshow(brances);

% br_prop=regionprops(brances,'PixelList');
npts_position=[];
count=1;
[columns rows] = find (brances);
wh_pts=[rows columns];

B=array2table(wh_pts,'VariableNames',{'x','y'});
for i=1:size_prop
    A=array2table(s1(i).PixelList,'VariableNames',{'x','y'});
    C2 = intersect(A,B)
    [len_pts , ~]=size(C2);
    if len_pts<6
        npts_position(count)=i;
        count=count+1;
    end
end
intpt=length(npts_position);
out=zeros(r,c);
for k=1:intpt
    len=length(s1(npts_position(k)).PixelList);
    if len>length(wh_pts)
        for i=1:len
            out(s1(npts_position(k)).PixelList(i+len),s1(npts_position(k)).PixelList(i))=255;
        end
    else
    end
end


figure,imshow(out);




% [l, n]=bwlabel(brances,8);
% bx=[];by=[];count=1;
% for i=1:n
%     for j=1:r
%         for k=1:c
%             if l(j,k)==i
%                 bx(i,count)=k;
%                 by(i,count)=j;
%                 count=count+1;
%             end
%         end
%     end
% end
% 
%==================Region growing======================
% Phi = segCroissRegion(30,img,65,18);
% figure,imshowpair(img,Phi,"montage");
% 
% function Phi = segCroissRegion(tolerance,Igray,x,y)
% if(x == 0 || y == 0)
%     imshow(Igray,[0 255]);
%     [x,y] = ginput(1);
% end
% Phi = false(size(Igray,1),size(Igray,2));
% ref = true(size(Igray,1),size(Igray,2));
% PhiOld = Phi;
% Phi(uint8(x),uint8(y)) = 1;
% while(sum(Phi(:)) ~= sum(PhiOld(:)))
%     PhiOld = Phi;
%     segm_val = Igray(Phi);
%     meanSeg = mean(segm_val);
%     posVoisinsPhi = imdilate(Phi,strel('disk',1,0)) - Phi;
%     voisins = find(posVoisinsPhi);
%     valeursVoisins = Igray(voisins);
%     Phi(voisins(valeursVoisins > meanSeg - tolerance & valeursVoisins < meanSeg + tolerance)) = 1;
% end
% 
% % Uncomment this if you only want to get the region boundaries
% % SE = strel('disk',1,0);
% % ImErd = imerode(Phi,SE);
% % Phi = Phi - ImErd;
% end
% 

