%  A demo for BLPForest
%
%   input:
%         bagsize:                    sizes of bags.
%         split.train_label:          label of the training data.
%         split.train_label:          label of the training data.
%         split.train_data_idx:       ID of each training data.
%         split.train_bag_idx         bag's ID of each training data.
%         split.train_bag_prop        proportions of each bags.
%         split.train_bag_prop_down   the lower bound of proportions of each bags.
%         split.train_bag_idx         the upper bound of proportions of each bags.
%         para.maxiter;               the maximum number of iterations of BLPForest.
%         para.bound_llp;             if para.bound_llp=1, BLPForest uses the interval of 
%                                     label proportions. Otherwise, BLPForest will degenerate
%                                     into a standard LLP problem.
%        N_random                     the number  of initialization.     
%
%   output:

%        result{j}.train_acc           the training accuracies of different initialization for some fixed size of bags.
%        result{j}.best_train_acc      the best accuracy in the  result.train_acc.
%        result{j}.mean_train_acc      the mean accuracy in the  result.train_acc.
%        result{j}.totaltime           the total training time of BLPForest. 
%        result{j}.obj = obj;         all object function value of   BLPForest for some fixed size of bags. 
%
%   Author: Zhiquan Qi
%   Date: 2016.01.05
%


clear;
para.maxiter = 100;
para.bound_llp=1;
bsize=[2 4 8 16 32 64];

filename = 'covtype_small.mat';
load(filename);
result = {};
for j=1:6
    rand('state',0);
    split = multi_classes_split_bags(split.train_label, bsize(j));
    class_num = max(split.train_label);
    para.class  = class_num;
    N_random = 20;
    train_acc = [];
    obj = zeros(N_random,1);
    train_len = length(split.train_data_idx);
    tic; t1=clock;
    for pp = 1:N_random
        para.init_y = ones(train_len,1);
        r = randperm(train_len);
        gg = floor(train_len/class_num);
        for k=1:class_num-1
            para.init_y(r(1+(k-1)*gg:gg*k)) = k;
        end
        para.init_y(r(gg*k+1:end))=k+1;
        re = LLPForestTrain(data, split, para);
        obj(pp)    = re.obj;
        train_acc = [train_acc re.train_acc];
    end
    toc;  t2=clock; totaltime=etime(t2,t1);
    [mm,id] = min(obj);
    best_train_acc = train_acc(id);
    result{j}.train_acc = train_acc;
    result{j}.best_train_acc = best_train_acc;
    result{j}.mean_train_acc = mean(train_acc);
    result{j}.totaltime  = totaltime;
    result{j}.obj = obj;
end






