I = imread('11.png');
BW = roipoly(I);
imshow(BW);
imwrite(BW,'11_mask.png')