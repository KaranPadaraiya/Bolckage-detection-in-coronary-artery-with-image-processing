
function newBlock = adjustContrast(blockStruct)
    % extract the image data for the block being processed 
    % from the structure blockStruct
    block = blockStruct.data;
    
    % rescale this block
    newBlock = rescale(block);
end

function [outImg]=gray2d(bw)
[r, c]=size(bw);
bw=imresize(bw,[r,c]);

 for x=1:3;
    for i=1:r
       for j=1:c
           outImg(i,j,x)=bw(i,j);
       end
    end
 end
end


function [outImg]=bi2gray(bw)
[r, c]=size(bw);

outImg=ones([r c]);
for i=1:r
    for j=1:c
        if bw(i,j)==1;
            outImg(i,j)=255;
        else
            outImg(i,j)=0;
        end
    end
end
end

function [outImg]=im21d(bw)
[r, c]=size(bw);

for i=1:r
    for j=1:c
        outImg((i-1)*c +j)=bw(i,j);
    end
end
end


function [outImg]=bpt(bw)
[l c]=size(bw);
x=1;
for i=1:c
    if bw(i)==0
        out(x)=i;
        x=x+1;
%     else
    end
end
outImg=out;
end

