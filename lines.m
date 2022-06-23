
i=imread("c21.jpg");
i=imresize(i,[50 70]);
[r c d]=size(i);
imtool(i)

x=1:c;
y=ones(1,c);
l1=[x;y];
l1=transpose(l1);

x=ones(1,r)*(c+1);
y=1:r;
l2=[x;y];
l2=transpose(l2);

y=ones(1,c)*(r+1);
x=1:c;
l3=[x;y];
l3=transpose(l3);

x=ones(1,r);
y=1:r;
l4=[x;y];
l4=transpose(l4);