videoReader = VideoReader("demo.avi");

frame = readFrame(videoReader);
frameGray = rgb2gray(frame);

prevPoints = detectSURFFeatures(frameGray, 'MetricThreshold', 500);
[prevFeatures, prevPoints] = extractFeatures(frameGray, prevPoints);

pose = eye(3);
trajectory = zeros(0, 2);

figure;
hold on;
axis equal;
title("Minimal Kamera Yolu");

while hasFrame(videoReader)
    frame = readFrame(videoReader);
    frameGray = rgb2gray(frame);

    points = detectSURFFeatures(frameGray, 'MetricThreshold', 500);
    if points.Count < 20
        continue;
    end

    [features, points] = extractFeatures(frameGray, points);

    indexPairs = matchFeatures(prevFeatures, features, 'MaxRatio', 0.6, 'Unique', true);
    matchedPoints1 = prevPoints(indexPairs(:, 1));
    matchedPoints2 = points(indexPairs(:, 2));

    if matchedPoints1.Count >= 8
        try
            tform = estimateGeometricTransform2D(matchedPoints1, matchedPoints2, "similarity", ...
                'MaxDistance', 4, 'Confidence', 95);

            deltaPose = eye(3);
            deltaPose(1:2, 1:2) = tform.T(1:2,1:2);
            deltaPose(1:2, 3) = tform.T(3,1:2);

            pose = pose * deltaPose;

            camPosition = pose(1:2, 3)';
            trajectory(end+1,:) = camPosition;

            if size(trajectory,1) > 1
                plot(trajectory(end-1:end,1), trajectory(end-1:end,2), 'b.-');
                drawnow;
            end
        catch
            disp("transform hatası geçildi");
        end
    end

    prevPoints = points;
    prevFeatures = features;
end

plot(trajectory(:,1), trajectory(:,2), 'r-', 'LineWidth', 2);
title("Kamera Yolu (Düzenlenmiş)");
