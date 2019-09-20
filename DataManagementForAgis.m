% Get LGE images as tif format and without cropping
clear all;
close all;

addpath('C:\Users\ZhangX1\Documents\MATLAB\cviParser\')
addpath('C:\Users\ZhangX1\Documents\MATLAB\InfarctDetector\')

base_dir = 'D:\Data\CNNData_ForAgis\';

name_glob = glob('D:\Data\CNNData\*');
sequence_label = {'LGE', 'T1'};
src_dir = 'D:\DeepLearn\src\';
CoordsFileName_T1 = [src_dir, sequence_label{2}, 'coords.csv'];
[num_t1,txt,raw_t1] = xlsread(CoordsFileName_T1);

%% Need to get the centroid first
%% Copy and crop T1 map infarct
interp_status = cell(length(name_glob), 3);
for i = 1:length(name_glob)
   T1_glob = glob(cat(2, name_glob{i}, 'T1\MI\MyoInfarct*.tif'));
   T1_centroid_glob = glob(cat(2, name_glob{i}, 'T1\Heart\masked_heart*.mat')); %mask_heart
   strings = split(name_glob{i}, '\');
   name = strings{end-1};
   if ~ exist(cat(2, base_dir, name), 'dir')
       mkdir(cat(2, base_dir, name));
   end
   
   if ~ exist(cat(2, base_dir, name, '\T1'), 'dir')
       mkdir(cat(2, base_dir, name, '\T1'));
   end
   
   if ~ exist(cat(2, base_dir, name, '\T1\MI\'), 'dir')
       t1_mi_dir = cat(2, base_dir, name, '\T1\MI\');
       mkdir(t1_mi_dir);
   end

   for j = 1:length(T1_glob)
       img_t1 = imread(T1_glob{j});
       strings = split(T1_glob{j}, '\');
       fname = strings{end};
       imwrite(img_t1, cat(2, t1_mi_dir, fname));
   end
   interp_status{i, 1} = name;
   interp_status{i, 2} = length(T1_glob);
   
   if i == 1
       interp_status{i, 3} = length(T1_glob);
   else
       interp_status{i, 3} = interp_status{i-1, 3} + length(T1_glob);
   end
end

%% Copy T1 map myocardium to dst
for i = 1:length(name_glob)
    T1_centroid_glob = glob(cat(2, name_glob{i}, 'T1\Heart\masked_heart*.mat'));
    strings = split(name_glob{i}, '\');
    name = strings{end-1};
    t1myo64_glob = glob(cat(2, name_glob{i}, 'T1\Myocardium\*'));
    out_dir = cat(2, base_dir, name, '\T1\Myocardium\');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir)
    end
    for j = 1:length(t1myo64_glob)
        strings = split(t1myo64_glob{j}, '\');
        fname = strings{end};
        strings = split(fname, '.');
        tif_name = cat(2, strings{1}, '.tif');
        load(t1myo64_glob{j})
        imwrite(mask_myocardium, cat(2, out_dir, tif_name))
    end
end

%% Find VOLUME_IMAGE and its corresponding slices
t1_org_dir = 'D:\Data\Patients_ForCNN\';
t1_glob = glob(cat(2, t1_org_dir, '*\T1\*\*\VOLUME_IMAGE.mat'));
count = 0;
for i = 1:length(t1_glob)
   strings = split(t1_glob{i}, '\');
   name = strings{end-4};
   
   out_dir = cat(2, base_dir, name, '\T1\img\');
   if ~exist(out_dir, 'dir')
        mkdir(out_dir)
   end
   t1myo_glob = glob(cat(2, base_dir, name, '\T1\Myocardium\*'));
   if ~isempty(t1myo_glob)
       
       dicom_idx = ExtractNum(t1myo_glob);
       load(t1_glob{i});
       if max(dicom_idx) > size(volume_image, 3)
           disp(name)
       else
           new_volume_image = volume_image(:,:,dicom_idx);
           for j = 1:length(dicom_idx)
               count = count + 1;
               mat_name = cat(2, num2str(count), '.mat');
               img = new_volume_image(:,:,j);
               save(cat(2, out_dir, mat_name), 'img')
           end
       end
   end
   
end
%% Find the corresponding layers & Copy LGE from cropped dir to dst 
addpath('C:\Users\ZhangX1\Documents\MATLAB\DeepLearn\')
sequence_label = {'LGE', 'T1'};
lge_glob = glob(cat(2, 'D:\Data\DataForWindowing\*\', sequence_label{1}, '\VOLUME_IMAGE.mat'));

t1_glob = glob(cat(2, 'D:\Data\DataForWindowing\*\', sequence_label{2}, '\VOLUME_IMAGE.mat'));
data_dir = 'D:\Data\CNNData\';
se = strel('square', 2);

for i = 1:length(lge_glob)
    strings = split(lge_glob{i}, '\');
    name = strings{4};
    lge = load(lge_glob{i});
    t1 = load(t1_glob{i});
    
    t1_myo_glob = glob(cat(2, base_dir, name, '\' , sequence_label{2}, '\Myocardium\*'));
    t1_idx = sort(GetGlobIndex(t1_myo_glob));
    
    LGE_centroid_glob = glob(cat(2, data_dir, name, '\LGE\Heart\masked_heart*.mat'));
    lge_myo_glob = glob(cat(2, data_dir, name, '\LGE\Myocardium\masked_myocardium*.mat'));
    lge_idx = sort(GetGlobIndex(LGE_centroid_glob));
    
    out_dir = cat(2, base_dir, name, '\LGE\');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir)
    end
    out_dir_img = cat(2, out_dir, 'img\');
    if ~exist(out_dir_img, 'dir')
        mkdir(out_dir_img)
    end
    out_dir_myo = cat(2, base_dir, name, '\LGE\Myocardium\');
    if ~exist(out_dir_myo, 'dir')
        mkdir(out_dir_myo)
    end
    
    out_dir_mi = cat(2, base_dir, name, '\LGE\MI\');
    if ~exist(out_dir_mi, 'dir')
        mkdir(out_dir_mi)
    end
    
    t1_cropped_glob = glob(cat(2, base_dir, name, '\T1\img\*.mat'));
    lge_mi_glob = glob(cat(2, data_dir, name, '\LGE\MI\*.tif'));
    % count = 1;
    if length(t1_idx) > length(lge_idx)
        t1_idx = intersect(t1_idx, lge_idx);
    end
    prev_t1_sloc = 0;
    prev_lge_sloc = 0;
    
    for j = 1:length(t1_idx)
        t1_sloc = t1.slice_data(t1_idx(j)).SliceLocation;
        t1_sloc = round(t1_sloc, 2);
        for k = 1:length(lge_idx)
            lge_sloc = lge.slice_data(lge_idx(k)).SliceLocation;
            lge_sloc = round(lge_sloc, 2);
            if round(abs(lge_sloc - t1_sloc), 2) <= 4
                break;
            end
        end
        if round(abs(lge_sloc - t1_sloc), 2) == 4
            % disp(k)
            if round((prev_t1_sloc + t1_sloc)/2, 2) == round(prev_lge_sloc, 2) || (round(abs(prev_t1_sloc - prev_lge_sloc), 2) < 4 && prev_t1_sloc ~= 0 && prev_lge_sloc ~= 0)
                k = k + 1;
                lge_sloc = lge.slice_data(lge_idx(k)).SliceLocation;
                lge_sloc = round(lge_sloc, 2);
                % disp(k)
            end
        end
        if round(abs(lge_sloc - t1_sloc), 2) <= 4
            prev_t1_sloc = t1_sloc;
            prev_lge_sloc = lge_sloc;
            % lge_idx = GetGlobIndex(LGE_centroid_glob);
            % if ~isempty(find(lge_idx == k, 1))
            correspond_lge = lge.volume_image(:, :, lge_idx(k));
            lge_heart = load(cat(2, data_dir, name, '\LGE\Heart\masked_heart', num2str(lge_idx(k)), '.mat'));
            % lge_myo
            lge_myo = load(cat(2, data_dir, name, '\LGE\Myocardium\masked_myocardium', num2str(lge_idx(k)), '.mat'));
            
            % lge_infarct
            lge_mi = imread(cat(2, data_dir, name, '\LGE\MI\MyoInfarct', num2str(k), '.tif'));
            
            mask_heart = lge_heart.mask_heart > 0;
            mask_myocardium = lge_myo.mask_myocardium > 0;
            
            strings = split(t1_cropped_glob{j}, '\');
            mat_name = strings{end};
            save(cat(2, out_dir_img, mat_name), 'correspond_lge')
            
            mask_myocardium = imclose(mask_myocardium, se);
            myo_name = cat(2, 'masked_myocardium', num2str(lge_idx(k)), '.tif');
            imwrite(mask_myocardium, cat(2, out_dir_myo, myo_name))
            
            mi_name = cat(2, 'MyoInfarct', num2str(k), '.tif');
            imwrite(lge_mi, cat(2, out_dir_mi, mi_name));
            %disp(count)
            disp(k)
            %count = count + 1;
            % disp(j)
            
        end
        
    end
end
%% Test and validate

length(glob(cat(2, base_dir, '\*\LGE\MI\')))
length(glob(cat(2, base_dir, '\*\LGE\MI\*')))
length(glob(cat(2, base_dir, '\*\LGE\Myocardium\*')))
length(glob(cat(2, base_dir, '\*\LGE\img\*.mat')))

length(glob(cat(2, base_dir, '\*\T1\MI\')))
length(glob(cat(2, base_dir, '\*\T1\MI\*')))
length(glob(cat(2, base_dir, '\*\T1\Myocardium\*')))
length(glob(cat(2, base_dir, '\*\T1\img\*.mat')))

for i = 1:length(lge_glob)
    strings = split(lge_glob{i}, '\');
    name = strings{end-2};
    mi_glob = glob(cat(2, base_dir, name, '\LGE\MI\*'));
    lg_glob = glob(cat(2, base_dir, name, '\LGE\*.mat'));
    if length(mi_glob) ~= length(lg_glob)
        disp(name)
    end
end