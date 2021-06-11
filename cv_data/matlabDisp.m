I1 = imread('~/cv_data/left/l_1.png');
I2 = imread('~/cv_data/right/r_1.png');
[R1, R2] = rectifyStereoImages(I1, I2, stereoParams, 'OutputView', 'full');
R2 = imtranslate(R2, [0,0]);
%figure
imshow(stereoAnaglyph(R1, R2))
R1 = rgb2gray(R1);
R2 = rgb2gray(R2);

disparityRange = [-64 64];
%disparityMap = disparitySGM(I1,I2,'DisparityRange',disparityRange,'UniquenessThreshold',20);
disparityMap = disparitySGM(R1,R2,'DisparityRange',disparityRange,'UniquenessThreshold',10);

figure
imshow(disparityMap, disparityRange)
title('Disparity Map')
colormap jet
colorbar

points3D = reconstructScene(disparityMap, stereoParams);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D);

% Create a streaming point cloud viewer
player3D = pcplayer([-100, 100], [-100, 100], [0, 100], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');
%colormap(player3D)
% Visualize the point cloud
view(player3D, ptCloud);

