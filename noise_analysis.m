A=imread('uniform.png');

A = imgaussfilt(A)
crop1=imcrop(A);
figure,imhist(crop1)

% crop2=imcrop(A);
% figure,imhist(crop2)
% 
% crop3=imcrop(A);
% figure,imhist(crop3)
% 
% crop4=imcrop(A);
% figure,imhist(crop4)
% 
[crop1,estDoS1] = imnlmfilt(crop1);
% [crop2,estDoS2] = imnlmfilt(crop2);
% [crop3,estDoS3] = imnlmfilt(crop3);
% [crop4,estDoS4] = imnlmfilt(crop4);

imhist(crop1);