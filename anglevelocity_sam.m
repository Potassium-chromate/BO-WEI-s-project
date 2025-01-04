clear;

d = uigetdir(pwd, 'Select a folder');
files = dir(fullfile(d, '*.tif'));

for i = 1:length(files)
    file_path = "沉沉291203/" + files(i).name;
    fprintf("Process %dth image: %s\n", i, files(i).name);
    CandA = Generate_mat(file_path);

    [~, file_name, ~] = fileparts(files(i).name); % Remove extension
    mat_name = fullfile("mat/", file_name + ".mat");

    % Save data to .mat file
    save(mat_name, 'CandA'); % 'CandA' must be passed as a string
end

function [CandA] = Generate_mat(file_path)
    image_x = 1000;
    image_y = 1000;
    Big_size = 24;                                                             
    Small_size = 18; 
    erodelevel = 5;
    
    Otsu = 0.1;                                                               % 調整內建閥值
    Threshold_G = 120;                                                         % 需調整邊界明顯與否   
    erode_level = 1;                                                           % 蝕刻將 Fringe 的雜訊消除
    Threshold_F = 0.5;                                                        % 找尋 Fringe 所需要的 threshold (比 otsu 高)
    Threshold_High = 0.74;
    STD_factor = 4.5;
    
    Oimg = imread(file_path);
    oimg_ocut = imcrop(Oimg,[0,0,image_x,image_y]);
    oimg_cut = imlocalbrighten(oimg_ocut,1);
    % imshow(oimg_cut);
    % imshow(Oimg);
    [r,c] = size(oimg_cut);
    
    
    %===========================底邊加入黑框===================================%
    for i = 1:c
        oimg_cut(r,i) = 0;
    end
    for i = 1:c
        oimg_cut(r-1,i) = 0;
    end
    for i = 1:c
        oimg_cut(r-2,i) = 0;
    end
    for i = 1:r
        oimg_cut(i,c) = 0;
    end
    for i = 1:r
        oimg_cut(i,1) = 0;
    end
    
    %========================== Wavelet Denoise ==============================%
    denoised = medfilt2(oimg_cut);                                             % 中值濾波
    %=========================================================================%
    [Gmag,~] = imgradient(denoised);                                           % 梯度與邊界
    denoised_g = medfilt2(Gmag);                                               % 邊界去雜訊
    otsu_gimg = imbinarize(denoised_g,Threshold_G);                            % 邊界二質化
    %imshow(otsu_gimg);
    se = strel('disk',2);                                
    otsu_gimg = imdilate(otsu_gimg,se);                                        % 邊界膨脹
    %imshow(otsu_gimg);
    %=========================================================================%
    CEimg = denoised;                                             % 調整對比度
    otsu = graythresh(CEimg);                                                  % Otsu threshold
    otsu_img = imbinarize(CEimg,otsu-Otsu);                                  % binary
    otsu_img_c = imcomplement(otsu_img);                                       % 黑白翻轉
    %imshow(otsu_img)
    %imshow(CEimg)
    
    
    
    % denoised1 = medfilt2(oimg_ocut);    
    % o_adapthisteq = adapthisteq(denoised1);
    % J = fibermetric(o_adapthisteq,7,"ObjectPolarity","bright","StructureSensitivity",5);
    % imshow(J);
    % H = im2uint8(J);
    % imshow(H);
    % 
    % oi = imbinarize(H,0.3); 
    % imshow(oi);
    % % se1 = strel('disk',3);
    % % oi = imdilate(oi,se1);
    % % imshow(oi);
    % 
    % % se1 = strel('disk', 1);
    % % oi_c = imerode(oi, se1);
    % % imshow(oi_c);
    % 
    % otsu_img_high = imbinarize(oimg_cut,otsu + Threshold_F);
    % imshow(otsu_img_high);
    % otsu_img_high_md = medfilt2(otsu_img_high);
    % imshow(otsu_img_high_md);
    % se1 = strel('disk', 8);
    % otsu_img_high_md = imdilate(otsu_img_high_md, se1);
    % imshow(otsu_img_high_md);
    % [label,~] = bwlabel(otsu_img_high_md);
    % %===================== 把已經抓到光條紋的變成黑色 =========================%
    % for x = 1:size(oi,1)
    %     for y = 1:size(oi,2)
    %         if label(x,y) ~= 0
    %                 oi(x,y) = 0;
    %         end
    %     end
    % end
    % imshow(oi);
    %===========================處理沒有顆粒的黑背景===========================%
    [labeled,N] = bwlabel(otsu_img_c,4);                                       % 標記 otsu_gimg
    for x = 1:r
        for y = 1:c
            if labeled(x,y)==1                                                 % 最大面積的黑底為背景
                CEimg(x,y) = 0;                                                % label = 1 → CEimg = 0
            end
        end
    end
    
    %imshow(CEimg);
    % Segmentation%%
    com = imcomplement(otsu_img);
    CEimg = double(CEimg);
    backG = CEimg.*com;
    a = find(com~=0);
    aa = size(a,1);
    mean1 = sum(backG(:))/aa; 
    bw7 = CEimg>=mean1;
    
    denoised_k = medfilt2(bw7);
    
    bw7 = denoised_k;
    %imshow(bw7);
    
    %%Take complement and labeling%%
    bw8 = imcomplement(bw7);
    [~,N] = bwlabel(bw8);
    CC = bwconncomp(bw8);
    bw9 = bw7;
    bw10 = imcomplement(bw9);
    CEimg = uint8(CEimg);
    
    %========================降低Threshold進行迭代============================%
    threshold_loop = otsu;
    while (N ~= 0) && (threshold_loop > (otsu/10))                             % Threshold 下限 !!!! 待調整
        threshold_loop = threshold_loop - 0.01;                                % 降低幅度 : 可用來減少計算時間
        bw9 = imbinarize(CEimg,threshold_loop);
        bw10 = imcomplement(bw9);
        [labeled,N] = bwlabel(bw10,4);
        CC = bwconncomp(bw10,4);
        By = bwboundaries(bw10,4,'noholes');
        BounG = zeros(N,4);
        for i = 1:N
            BounG(i,1) = size(By{i},1);
            count = 0;
            for j = 1:BounG(i,1)
                if otsu_gimg(By{i}(j,1),By{i}(j,2)) == 1
                    count = count+1;
                end
            end
            BounG(i,2) = count;
            BounG(i,3) = BounG(i,2)/BounG(i,1);
            BounG(i,4) = size(CC.PixelIdxList{i},1);
        end
        
        for i = 1:N
            if BounG(i,3) < 0.2 && BounG(i,4) < 3
                bw9(CC.PixelIdxList{i}) = 1;
            end
            
        end
        CEimg(bw9) = 255;
        
        BGsure = find(BounG(:,3)>=0.98);
        BGsr = size(BGsure,1);
        for i = 1:BGsr
            CEimg(labeled == BGsure(i)) = 0;
        end
        [~,N] = bwlabel(CEimg,4);                                              % 計算剩餘需計算區域
    end
    %imshow(CEimg);
    Icropped_sobel =edge(Oimg,'prewitt');
    %imshow(Icropped_sobel);
    
    CEimg1 = xor(CEimg, Icropped_sobel);
    %imshow(CEimg1);
    
    otsu_img_high = imbinarize(oimg_cut,otsu + Threshold_F);
    %imshow(otsu_img_high);
    otsu_img_high_md = medfilt2(otsu_img_high);
    %imshow(otsu_img_high_md);
    se1 = strel('disk', 6);
    otsu_img_high_md = imdilate(otsu_img_high_md, se1);
    %imshow(otsu_img_high_md);
    [label,~] = bwlabel(otsu_img_high_md);
    %===================== 把已經抓到光條紋的變成黑色 =========================%
    for x = 1:size(CEimg1,1)
        for y = 1:size(CEimg1,2)
            if label(x,y) ~= 0
                    CEimg1(x,y) = 1;
            end
        end
    end
    %imshow(CEimg1);
    
    % f = ~CEimg1;
    % imshow(f);
    % stats = regionprops(f, 'Area');
    % 
    % % 获取所有区域的面积
    % allAreas = [stats.Area];
    % 
    % % 创建一个新的二值图像，初始化为全零
    % filteredImage = zeros(size(f));
    % 
    % % 筛选出面积大于等于30的区域
    % for k = 1:length(allAreas)
    %     if allAreas(k) >= 10
    %         % 获取当前区域的像素索引
    %         currentRegion = ismember(bwlabel(f), k);
    %         % 将这些像素添加到新图像中
    %         filteredImage = filteredImage | currentRegion;
    %     end
    % end
    % 
    % % 显示结果
    % imshow(filteredImage);
    % 
    % CEimg1 = ~filteredImage;
    
    %=======================小幅度蝕刻避免顆粒過於黏合=========================%
    se1 = strel('disk',2);        
    erode = imerode(CEimg1,se1);
    erode = logical(erode);
    %imshow(erode);
    
    %=======================切割兩圓連接處幫助蝕刻=============================%
    D = -bwdist(~erode);
    maskk = imextendedmin(D,2);
    D2 = imimposemin(D,maskk);
    Ld2 = watershed(D2);
    erode3 = erode;
    erode3(Ld2 == 0) = 0;
    se1 = strel('disk',1);        
    erode = imerode(erode3,se1);
    erode = imfill(erode,'holes');
    %imshow(erode);
    %===========================大幅度蝕刻取質心==============================%
    se1 = strel('disk',erodelevel);                                                    
    ER = imerode(erode,se1);
    % ER = imclearborder(ER,8);
    % se = strel('disk',3);                                
    % ER = imdilate(ER,se);
    %imshow(ER);
    stats = regionprops(ER, 'Area');
    
    % 获取所有区域的面积
    allAreas = [stats.Area];
    
    % 创建一个新的二值图像，初始化为全零
    filteredImage = zeros(size(ER));
    
    % 筛选出面积大于等于30的区域
    for k = 1:length(allAreas)
        if allAreas(k) >= 100
            % 获取当前区域的像素索引
            currentRegion = ismember(bwlabel(ER), k);
            % 将这些像素添加到新图像中
            filteredImage = filteredImage | currentRegion;
        end
    end
    
    % 显示结果
    % imshow(filteredImage);
    % filteredImage = ER;
    [Label,NN] = bwlabel(ER);
    CandA = regionprops(Label,'centroid','Area');
    centroid = cat(1,CandA.Centroid);
    area = cat(1,CandA.Area);
    CandA = cat(2,centroid,area);
    
    %============================判斷粒徑大小==================================%
    for i = 1:NN
        if (CandA(i,3) >= 700) && (CandA(i,3) < 2300)
            CandA(i,3) = Big_size;
        elseif (CandA(i,3) < 700) && (CandA(i,3) >= 100)
            CandA(i,3) = Small_size;
        else 
            CandA(i,:) = 0;
        end
    end
    %=========================去除蝕刻後面積太小的=============================%
    n = 1;
    error = 0;
    while NN-error >= n
        if CandA(n,:) == 0
           CandA(n,:) = [];
           error = error + 1;
        else
            n = n + 1;
        end
    end
    NN = NN - error;
    %==================== Remove the overlapping disk ========================%
    distance_error = 8;
    for i = 1:NN-1
        for j = i+1:NN
             ddx = CandA(i,1)-CandA(j,1);
             ddy = CandA(i,2)-CandA(j,2);
             DD = sqrt(ddx^2+ddy^2);                                           % 兩圓實際距離
             DDD = CandA(i,3)+CandA(j,3);                                      % 兩圓半徑和
             if DD < DDD - distance_error                                      % overlapping 
                 CandA(i,:) = 0;
             end
        end
    end
    n = 1;
    error = 0;
    while NN-error >= n
        if CandA(n,:) == 0
           CandA(n,:) = [];
           error = error + 1;
        else
            n = n + 1;
        end
    end
    NN = NN - error;
    
    % figure
    % imshow(oimg_cut)
    % cir_Small = CandA(CandA(:,3)==Small_size,:);
    % cir_Big   = CandA(CandA(:,3)==Big_size,:);
    % viscircles(cir_Small(:,1:2), cir_Small(:,3),'Color','b');
    % viscircles(cir_Big(:,1:2), cir_Big(:,3),'Color','r');
    % hold on;
    % plot(CandA(:,1),CandA(:,2),'y+','MarkerSize',6);


    [centers, radius, metric] = imfindcircles(erode,[13 26],'Sensitivity',0.9,'Method','TwoStage');
    
    centersStrong = centers(:,:); 
    radiiStrong = radius(:);
    metricStrong = metric(:);
    
    % 圓的數量
    numCircles = size(centersStrong, 1);
    
    % 找出重疊的圓組
    overlappingGroups = cell(numCircles, 1); % 用於儲存重疊圓的組別
    
    for i = 1:numCircles
        for j = i+1:numCircles
            % 獲取圓 i 和圓 j 的參數
            x1 = centersStrong(i, 1);
            y1 = centersStrong(i, 2);
            r1 = radiiStrong(i);
            x2 = centersStrong(j, 1);
            y2 = centersStrong(j, 2);
            r2 = radiiStrong(j);
    
            % 計算兩圓心之間的距離
            d = sqrt((x2 - x1)^2 + (y2 - y1)^2);
    
            % 檢查是否重疊（質心距離小於半徑和）
            if d < (r1 + r2-5)
                overlappingGroups{i} = [overlappingGroups{i}, j]; % 將重疊的圓加入組別
                overlappingGroups{j} = [overlappingGroups{j}, i]; % 雙向記錄
            end
        end
    end
    
    % 初始化一個要刪除的圓列表
    circlesToDelete = [];
    
    % 對每個重疊組計算接觸數量，找出接觸數量最多的圓
    for i = 1:numCircles
        if ~isempty(overlappingGroups{i})
            % 取得重疊的圓組（包括自己）
            group = unique([i, overlappingGroups{i}]);
            contactCounts = zeros(length(group), 1);
    
            % 計算每個圓的接觸數量
            for k = 1:length(group)
                currentCircle = group(k);
                x1 = centersStrong(currentCircle, 1);
                y1 = centersStrong(currentCircle, 2);
                r1 = radiiStrong(currentCircle);
                
                for j = 1:numCircles
                    if j ~= currentCircle
                        x2 = centersStrong(j, 1);
                        y2 = centersStrong(j, 2);
                        r2 = radiiStrong(j);
                        d = sqrt((x2 - x1)^2 + (y2 - y1)^2);
    
                        % 判斷接觸點（質心距離 <= 半徑和）
                        if d <= (r1 + r2)
                            contactCounts(k) = contactCounts(k) + 1;
                        end
                    end
                end
            end
    
            % 找出接觸數量最多的圓，標記為要刪除的圓
            [~, idxToDelete] = max(contactCounts);
            circlesToDelete = [circlesToDelete, group(idxToDelete)];
        end
    end
    
    % 刪除重複的圓（避免多次刪除同一圓）
    circlesToDelete = unique(circlesToDelete);
    
    % 刪除錯誤的圓
    centersStrong(circlesToDelete, :) = [];
    radiiStrong(circlesToDelete) = [];
    metricStrong(circlesToDelete) = [];
    
    % 顯示刪除後的圓
    % figure;
    % imshow(oimg_cut);
    % viscircles(centersStrong, radiiStrong, 'EdgeColor', 'b');


    centersStrong_num = size(centersStrong,1);
    for i = 1:centersStrong_num
            distances = sqrt((centersStrong(i, 1) - CandA(:, 1)).^2 + (centersStrong(i, 2) - CandA(:, 2)).^2);
            hough_c = min(distances);
             [~, idx] = min(distances);
            
            % 比較半徑差異
            radius_difference = abs(radiiStrong(i, 1) - CandA(idx, 3)); % 假設CandA的半徑在第3列
    
            if hough_c>4 || radius_difference > 5%
                CandA = [CandA; [centersStrong(i, 1), centersStrong(i, 2), radiiStrong(i, 1)]]; % 添加中心點和半
           
            end
    end
    
    CandA_num = size(CandA,1);
    for j = 1:CandA_num
        if CandA(j,3)<20
           CandA(j,3)=18;
        else
            CandA(j,3)=24;
        end
    end
    % figure
    % imshow(oimg_cut)
    % cir_Small = CandA(CandA(:,3)==Small_size,:);
    % cir_Big   = CandA(CandA(:,3)==Big_size,:);
    % viscircles(cir_Small(:,1:2), cir_Small(:,3),'Color','b');
    % viscircles(cir_Big(:,1:2), cir_Big(:,3),'Color','r');
    % hold on;
    % plot(CandA(:,1),CandA(:,2),'y+','MarkerSize',6);
    % 圓的數量
    numCircles = size(CandA, 1);
    
    % 找出重疊的圓組
    overlappingGroups = cell(numCircles, 1); % 用於儲存重疊圓的組別
    
    for i = 1:numCircles
        for j = i+1:numCircles
            % 獲取圓 i 和圓 j 的參數
            x1 = CandA(i, 1);
            y1 = CandA(i, 2);
            r1 = CandA(i,3);
            x2 = CandA(j, 1);
            y2 = CandA(j, 2);
            r2 = CandA(j,3);
    
            % 計算兩圓心之間的距離
            d = sqrt((x2 - x1)^2 + (y2 - y1)^2);
    
            % 檢查是否重疊（質心距離小於半徑和）
            if d < (r1 + r2-7)
                overlappingGroups{i} = [overlappingGroups{i}, j]; % 將重疊的圓加入組別
                overlappingGroups{j} = [overlappingGroups{j}, i]; % 雙向記錄
            end
        end
    end
    
    % 初始化一個要刪除的圓列表
    circlesToDelete = [];
    
    % 對每個重疊組計算接觸數量，找出接觸數量最多的圓
    for i = 1:numCircles
        if ~isempty(overlappingGroups{i})
            % 取得重疊的圓組（包括自己）
            group = unique([i, overlappingGroups{i}]);
            contactCounts = zeros(length(group), 1);
    
            % 計算每個圓的接觸數量
            for k = 1:length(group)
                currentCircle = group(k);
                x1 = CandA(currentCircle, 1);
                y1 = CandA(currentCircle, 2);
                r1 = CandA(currentCircle,3);
                
                for j = 1:numCircles
                    if j ~= currentCircle
                        x2 = CandA(j, 1);
                        y2 = CandA(j, 2);
                        r2 = CandA(j,3);
                        d = sqrt((x2 - x1)^2 + (y2 - y1)^2);
    
                        % 判斷接觸點（質心距離 <= 半徑和）
                        if d <= (r1 + r2-5)
                            contactCounts(k) = contactCounts(k) + 1;
                        end
                    end
                end
            end
    
            % 找出接觸數量最多的圓，標記為要刪除的圓
            [~, idxToDelete] = max(contactCounts);
            circlesToDelete = [circlesToDelete, group(idxToDelete)];
        end
    end
    
    % 刪除重複的圓（避免多次刪除同一圓）
    circlesToDelete = unique(circlesToDelete);
    
    % 刪除錯誤的圓
    CandA(circlesToDelete, :) = [];
    
    % figure
    % imshow(oimg_cut)
    % cir_Small = CandA(CandA(:,3)==Small_size,:);
    % cir_Big   = CandA(CandA(:,3)==Big_size,:);
    % viscircles(cir_Small(:,1:2), cir_Small(:,3),'Color','b');
    % viscircles(cir_Big(:,1:2), cir_Big(:,3),'Color','r');
    % hold on;
    % plot(CandA(:,1),CandA(:,2),'y+','MarkerSize',6);

    %刪除左右兩邊超過圖範圍的圓
    num_c = size(CandA,1);
    rowsToDelete = [];
    for i = 1: num_c;
        if CandA(i,1)+CandA(i,3)>1000 || CandA(i,1)-CandA(i,3)<0
            rowsToDelete = [rowsToDelete; i]; % 記錄需要刪除的行索引
        end
    end
    
    CandA(rowsToDelete, :) = []; % 刪除滿足條件的行
    
    % figure
    % imshow(oimg_cut)
    % cir_Small = CandA(CandA(:,3)==Small_size,:);
    % cir_Big   = CandA(CandA(:,3)==Big_size,:);
    % viscircles(cir_Small(:,1:2), cir_Small(:,3),'Color','b');
    % viscircles(cir_Big(:,1:2), cir_Big(:,3),'Color','r');
    % hold on;
    % plot(CandA(:,1),CandA(:,2),'y+','MarkerSize',6);
end