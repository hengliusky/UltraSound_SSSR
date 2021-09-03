clear;close all;

folder = './';
scale = 4;

%% generate data
filepaths = dir(fullfile(folder,'*.jpg'));
for i = 1 : length(filepaths)
    image = imread(fullfile(folder,filepaths(i).name));
    im_label = modcrop(image, scale);
    im_lr= imresize(im_label, 1/scale, 'bicubic');
    imwrite(im_lr, ['./lr/', num2str(i), '.png'])
    imwrite(im_label,['./label/', num2str(i), '.png']);
end

function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end
end