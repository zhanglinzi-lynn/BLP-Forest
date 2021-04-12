function split = multi_classes_split_bags(label, bagsize)

%   split = multi_classes_split_bags(label, bagsize)
%   input:
%           label:               label of the training data.
%           bagsize:             sizes of bags.
%
%   output:
%         split.train_label:          label of the training data.
%         split.train_data_idx:       ID of each training data.
%         split.train_bag_idx         bag's ID of each training data.
%         split.train_bag_prop        proportions of each bags.
%         split.train_bag_prop_down   the lower bound of proportions of each bags.
%         split.train_bag_idx         the upper bound of proportions of each bags.

%
%   Author: Zhiquan Qi
%   Date: 2016.04.06
%


class_num = max(label);
len = length(label);
bagnum = ceil(len/bagsize) ;

split.train_label = label;
split.train_data_idx = 1:len;
split.train_bag_idx = zeros(len, 1);
for i=1:len,
    split.train_bag_idx(i) = mod(i, bagnum);
end
split.train_bag_idx(split.train_bag_idx == 0) = bagnum;
rowrank = randperm(size(split.train_bag_idx, 1));
split.train_bag_idx = split.train_bag_idx(rowrank, :);

split.train_bag_prop = zeros(bagnum, class_num);
for i=1:bagnum,
    for l =1:class_num,
        sa = split.train_label == l;
        sb = split.train_bag_idx == i;
        split.train_bag_prop(i,l) = sum(sa+sb == 2);
        
        split.train_bag_prop(i,l) = split.train_bag_prop(i,l)/length(find(split.train_bag_idx==i));
    end
end
% bounded constraint
range = 0.1;
split.train_bag_prop_down = max(0,split.train_bag_prop-range);
split.train_bag_prop_up   = min(1,split.train_bag_prop+range);
end

