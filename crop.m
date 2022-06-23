A=imread('rankfiltered.jpg');
% [outImg]=gray2d(bw)

%{
================================================
noisyLAB = rgb2lab(i);
roi = [210,24,52,41];
patch = imcrop(noisyLAB,roi);
patchSq = patch.^2;
edist = sqrt(sum(patchSq,3));
patchSigma = sqrt(var(edist(:)));
DoS = 1.1*patchSigma;
denoisedLAB = imnlmfilt(noisyLAB,'DegreeOfSmoothing',DoS);
denoisedRGB = lab2rgb(denoisedLAB,'Out','uint8');
imshow(denoisedRGB)


grimg=rgb2gray(denoisedRGB);
x=adapthisteq(grimg, 'numTiles', [8 8], 'nBins', 128);
imshow(x);


Test_Image=gray2d(denoisedRGB);
Resized_Image = imresize(Test_Image, [512 512]);
Converted_Image = im2double(Test_Image);
Lab_Image = rgb2lab(Resized_Image);
fill = cat(3, 1,0,0);
Filled_Image = bsxfun(@times, fill, Lab_Image);
Reshaped_Lab_Image = reshape(Filled_Image, [], 3);
[C, S] = pca(Reshaped_Lab_Image);
S = reshape(S, size(Lab_Image));
S = S(:, :, 1);

Gray_Image = (S-min(S(:)))./(max(S(:))-min(S(:)));

Enhanced_Image = adapthisteq(Gray_Image, 'numTiles', [8 8], 'nBins', 128);

figure, imshow(Enhanced_Image)
title('Enhanced_Image')

Avg_Filter = fspecial('average', [9 9]);
Filtered_Image = imfilter(Enhanced_Image, Avg_Filter);
figure, imshow(Filtered_Image)
title('Filtered Image')

Substracted_Image = imsubtract(Filtered_Image,Enhanced_Image);
figure, imshow(Substracted_Image)
title('Substracted image')



%================================================
%}

ig=im2gray(A);
[r,c,l]=size(ig);
x=8;
nr=r/x;
nc=c/x;
skip_x=nr/4; %32
skip_y=nc/4; %32

new=zeros(r,c);
count=0;
for i=0:skip_x:r-nr
    for j=0:skip_y:c-nc
        count=count+1;
        cropped_region =imcrop(ig,[j i nc nr]);
%         cropped_region=adapthisteq(cropped_region, 'numTiles', [8 8], 'nBins', 128);
        str=string("croped/"+count+"image_orig.bmp");
        imwrite(cropped_region,str);
        cropped_region=vessel_seg(cropped_region);
        str=string("croped/"+count+"image.bmp");
        imwrite(cropped_region,str);
        for k=1:nc
            for l=1:nr
                new(k+i,l+j)=cropped_region(k,l) | new(k+i,l+j);
            end
        end
        imshow(new);

    end
end

ne=bwareaopen(new,2000);
figure,imshow(ne)

function [outImg]=gray2d(bw)
[r, c,l]=size(bw);
bw=imresize(bw,[r,c]);
if l==3
    outImg=bw;
else
  for x=1:3
    for i=1:r
       for j=1:c
           outImg(i,j,x)=bw(i,j);
       end
    end
 end
end

end

% convex 