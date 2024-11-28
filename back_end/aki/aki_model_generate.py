# 大概是可以跑通我们的数据集了，即ards_processed_data.npy
# 修改了一些模型的参数，本文件注释为中文的地方都是我修改的地方
# 这里还有几个需要解决的问题：
# 1. 模型输出的理解，即我们需要通过模型的输出得到规则
#  relation中含有谓词组合的时间关系，A矩阵选出谓词组合
# 2. 我现在需修改的参数只是让模型可以跑通，最后其他参数的设置还需要研究
# 3. 模型初始化时self.num_formula的设置，我们要依照什么？还需要研究，我这里先设置为2
#  需要生成多少条规则，就设置规则数+1条
#  python /home/yunyang2/EventSequenceClustering/aki/aki-Step1.py
import numpy as np
import itertools
# from clustering_generate_data_v1 import Logic_Model_Generator
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import csv
import datetime
import math
import time
torch.manual_seed(2)
# from utils import redirect_log_file, Timer

##################################################################
class Logic_Model(nn.Module): 
    # Assumption: body predicates: X1 ~ X5;  head predicate: X6
    # rules(related to X1~X5): 
    # f0: background; 
    # f1: X1 AND X2 AND X3, X1 before X2; 
   
    def __init__(self):
        super(Logic_Model, self).__init__()  # 确保调用父类的构造函数
        ### the following parameters are used to manually define the logic rules
        # 原先有10条体谓词，1个头谓词，设置为11
        # 现在有37条体谓词，1个头谓词，设置为38
        self.num_predicate = 38
        # 我这里先设置为2
        self.num_formula = 11

        self.body_predicate_set = list(np.arange(0,(self.num_predicate-1),1)) 
        self.head_predicate_set = [self.num_predicate-1] 
        
        self.empty_pred = 2 # append 2 empty predicates
        self.k = 2 # relaxedtopk parameter topk 
        self.sigma = 0.1 # laplace kernel parameter
        self.temp = 0.15 # 0.05,0.01, 0.15
        self.tolerance = 0.1
        self.prior = torch.tensor([0.01,0.99], dtype=torch.float64, requires_grad=True)
        
        
        self.weight = (torch.ones((self.num_formula-1), (len(self.body_predicate_set)+self.empty_pred)) * 0.000000001).double()
        self.weight = F.normalize(self.weight, p=1, dim = 1)
        self.weight.requires_grad = True

        self.relation = {}
        for i in range(self.num_formula-1):
            self.relation[str(i)] = {}
            
        self.model_parameter = {}
        head_predicate_idx = self.head_predicate_set[0]
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.ones(1)*0.02 
        self.model_parameter[head_predicate_idx]['weight'] = torch.autograd.Variable((torch.ones(self.num_formula - 1) * 0.5).double(), requires_grad=True)


    def log_p_star(self, head_predicate_idx, t, pi, data_sample, A, add_relation = True, relation_grad = True):
        # background
        cur_intensity = self.model_parameter[head_predicate_idx]['base']
        log_p_star_0 = torch.log(cur_intensity.clone()) + (- t * cur_intensity) + torch.log(pi[0])
        body_predicate_indicator = data_sample <= t
        body_predicate_indicator = torch.cat([body_predicate_indicator,torch.ones(len(t),self.empty_pred)],dim=1)
        body_predicate_indicator = body_predicate_indicator.repeat(1, A.shape[0])
        body_predicate_indicator = body_predicate_indicator.reshape(len(t), A.shape[0], A.shape[1])

        # use laplace kernel to calculate feature    
        feature_formula = torch.exp(-abs(torch.sum(body_predicate_indicator * A, dim=2) - self.k) / self.sigma)
        relation_used = {}
        if add_relation:
            topk_idx = A.sort(1,descending=True)[1][:,:self.k]
            topk_idx = topk_idx.sort(1,descending=False)[0]
            topk_val = A.sort(1,descending=True)[0][:,:self.k]
            relation_features = []

            # 修改张量大小
            relation_feature = torch.ones(7,self.num_formula-1)
            for i in range(self.num_formula-1): # iterates rules
                relation_used[str(i)]=[]
                rule_relation_features = []
                idx = topk_idx[i,:]
                # find body predicates, exclude dummy variables
                select_body = list(idx.numpy() < (self.num_predicate-1))
                if select_body[-1] == 0:
                    body_idx = idx[:select_body.index(0)]
                else:
                    body_idx = idx

                if len(body_idx) > 1: # only when the num of body pred larger than 1 , there will be temporal relation
                    body_idx2 = (body_idx.repeat(1,2)).reshape(2,-1)
                    # print(f"体谓词重复并 reshape 后的 body_idx2:\n{body_idx2}")
                    idx_comb = np.array(list(itertools.product(*body_idx2)))
                    # find k(k-1)/2 combination for a rule
                    idx_comb = np.delete(idx_comb, np.arange(0, len(body_idx)**2, len(body_idx)+1), axis=0)
                    #for k in range(idx_comb.shape[0]):
                    #    relation_used[str(i)].append(str(list(idx_comb[k, :])))
                    #    print(f"规则 {i} 的体谓词组合: {idx_comb[k, :]}")
                    delete_set = []
                    for j in range(len(body_idx)-1):
                        delete_set = delete_set + list(np.arange(j+(len(body_idx)-1)*(j+1), len(body_idx)**2-len(body_idx), len(body_idx)-1))
                    idx_comb = np.delete(idx_comb, delete_set, axis=0)
                    # get prob from the dict and time indicator
                    for k in range(idx_comb.shape[0]):
                        relation_used[str(i)].append(str(list(idx_comb[k,:])))
                        if str(list(idx_comb[k,:])) in self.relation[str(i)]:
                            self.relation[str(i)][str(list(idx_comb[k,:]))].requires_grad = True
                            prob = self.relation[str(i)][str(list(idx_comb[k,:]))]
                        else:
                            self.relation[str(i)][str(list(idx_comb[k,:]))] = F.normalize(torch.ones(4)*torch.tensor([1,1,1,10]), p=1, dim = 0)
                            self.relation[str(i)][str(list(idx_comb[k,:]))].requires_grad = True
                            prob = self.relation[str(i)][str(list(idx_comb[k,:]))]
                        
                        time_diff = data_sample[:,idx_comb[k,0]] - data_sample[:,idx_comb[k,1]]
                        time_binary_indicator = torch.zeros(len(time_diff),4)
                        time_binary_indicator[:,0] = time_diff > self.tolerance # after
                        time_binary_indicator[:,1] = abs(time_diff) < self.tolerance # equal
                        time_binary_indicator[:,2] = time_diff < -self.tolerance # before
                        time_binary_indicator[:,3] = 1-prob[0]*time_binary_indicator[:,0].clone()-prob[1]*time_binary_indicator[:,1].clone()-prob[2]*time_binary_indicator[:,2].clone() 

                        rule_relation_features.append(self.softmax(time_binary_indicator * prob))   
                else:
                    continue
                rule_relation_feature = self.softmin(torch.stack(rule_relation_features,dim=1))
                relation_feature[:,i] = rule_relation_feature
            
            feature_formula = feature_formula * relation_feature

        # get soft max body time (our method)
        data_sample = data_sample.repeat(1, A.shape[0])
        data_sample = data_sample.reshape(len(t), A.shape[0], -1)
        max_body_time = torch.max(body_predicate_indicator[:,:,0:(self.num_predicate-1)] * data_sample[:,:,0:(self.num_predicate-1)] * A[:,0:(self.num_predicate-1)], dim=2)[0] 
        
        t = t.repeat(1, A.shape[0])
        t = t.reshape(len(t),A.shape[0])
        sigm = torch.sigmoid((t - max_body_time)) # max body time < t: 1;   max body time > t: 0
        pre_intensity = self.model_parameter[head_predicate_idx]['base']
        cur_intensity = self.model_parameter[head_predicate_idx]['base'] + sigm * feature_formula * self.model_parameter[head_predicate_idx]['weight'] 
        log_p_star = torch.log(cur_intensity.clone()) + (- t * cur_intensity + sigm * (- max_body_time * pre_intensity + max_body_time * cur_intensity)) + torch.log(pi[1:])
        return torch.cat([log_p_star_0, log_p_star],dim=1), relation_used

    def softmin(self, x):
        exp_x = torch.exp(-x/self.temp)
        return torch.sum(exp_x * x, dim = 1) / torch.sum(exp_x, dim = 1)
    
    def softmax(self, x):
        exp_x = torch.exp(x/self.temp)
        return torch.sum(exp_x * x, dim = 1) / torch.sum(exp_x, dim = 1)

    def reparametrization(self, avg_num, tau):
        weight = self.weight.expand(avg_num,self.weight.shape[0],self.weight.shape[1])
        EPSILON = torch.from_numpy(np.array(np.finfo(np.float32).tiny))
        scores = torch.log(weight.clone())
        # print(self.weight)
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g
        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, EPSILON)
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / tau, dim=2)
            khot = khot + onehot_approx
        A =  khot
        return torch.mean(A, axis=0)

    def optimize_EM_single(self, body_time, head_time, batch_size, head_predicate_idx, optimizer, pi, tau, num_samples):
        qz = [] # E step
        num_batch = num_samples / batch_size
        EPSILON = torch.from_numpy(np.array(np.finfo(np.float32).tiny))

        dict = {}
        # 1. update rule content： self.weight and relation
        self.model_parameter[head_predicate_idx]['weight'].requires_grad = False
        
        n=1
        for i in range(n):
            indices = np.arange(body_time.shape[0])
            np.random.shuffle(indices)
            body_time = body_time[indices,:]
            head_time = head_time[indices]
            
            for batch_idx in np.arange(0, num_batch, 1): # iterate batches
                # annealing
                if batch_idx % 10 == 0:
                    tau = torch.max(tau / 2, torch.ones(1) * 0.1)
                sample_ID_batch = np.arange(batch_idx*batch_size, (batch_idx+1)*batch_size, 1)
                self.weight.requires_grad = True
                optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
                avg_num = 10
                A = self.reparametrization(avg_num, tau)
                # iterate over a batch
                data_sample = body_time[sample_ID_batch,:]
                t = head_time[sample_ID_batch,:]
                log_p_star, relation_used = self.log_p_star(head_predicate_idx, t, pi, data_sample, A)
                log_avg = log_p_star          # batch_size x num rules 
                qzi = F.normalize(torch.exp(log_avg),p=1,dim=1)
                log_likelihood = torch.sum(qzi * log_avg) 
                # 通过熵约束增加显著性
                # 损失函数中添加熵约束
                relation_entropy = 0
                for k, v in relation_used.items():
                    for predicate_combination in v:
                        if isinstance(self.relation[k][predicate_combination], torch.Tensor):
                            prob = self.relation[k][predicate_combination]
                            relation_entropy += torch.sum(prob * torch.log(prob + 1e-8))
                loss = -log_likelihood +0.1 * relation_entropy
                # loss = -log_likelihood  
                loss.backward(retain_graph=True)
                # update weight
                with torch.no_grad():
                    grad_Weight = self.weight.grad
                    self.weight -= grad_Weight * 0.0001
                    self.weight = torch.maximum(self.weight, EPSILON)
                    self.weight = torch.minimum(self.weight, torch.from_numpy(np.array(np.finfo(np.float32).max)))
                    self.weight = F.normalize(self.weight, p=1, dim = 1)
                   

        m = 1
        for i in range(m):
            np.random.shuffle(indices)
            body_time = body_time[indices,:]
            head_time = head_time[indices]
            
            for batch_idx in np.arange(0, num_batch/2, 1): # iterate batches
                tau = torch.ones(1)*0.1
                sample_ID_batch = np.arange(batch_idx*batch_size, (batch_idx+1)*batch_size, 1)
                self.weight.requires_grad = True
                optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
                avg_num = 10
                A = self.reparametrization(avg_num, tau)
                # iterate over a batch
                data_sample = body_time[sample_ID_batch,:]
                t = head_time[sample_ID_batch,:]
                # calculate expectation               return matrix: batch_size x num rules
                log_p_star, relation_used = self.log_p_star(head_predicate_idx, t, pi, data_sample, A)
                log_avg = log_p_star          # batch_size x num rules 
                qzi = F.normalize(torch.exp(log_avg),p=1,dim=1)
                log_likelihood = torch.sum(qzi * log_avg) 
                # 通过熵约束增加显著性
                # 损失函数中添加熵约束
                relation_entropy = 0
                for k, v in relation_used.items():
                    for predicate_combination in v:
                        if isinstance(self.relation[k][predicate_combination], torch.Tensor):
                            prob = self.relation[k][predicate_combination]
                            relation_entropy += torch.sum(prob * torch.log(prob + 1e-8))
                loss = -log_likelihood +0.1 * relation_entropy
                # loss = -log_likelihood  
                loss.backward(retain_graph=True)
                
                with torch.no_grad():
                    # update relation
                    for k,v in relation_used.items():
                        if len(v) > 0:
                            for j in range(len(v)):
                                grad_relation = self.relation[k][v[j]].grad
                                self.relation[k][v[j]] -= grad_relation * 0.35 #0.25，0.3， 0.35
                                # 这里做了修改，测试更大的步长
                                self.relation[k][v[j]] = torch.maximum(self.relation[k][v[j]], EPSILON)
                                self.relation[k][v[j]] = F.normalize(self.relation[k][v[j]], p=1, dim=0)
        
        # 2. update model parameters: self.model_parameter[head_predicate_idx]['weight']
        
        indices = np.arange(body_time.shape[0])
        np.random.shuffle(indices)
        body_time = body_time[indices,:]
        head_time = head_time[indices]

        self.weight.requires_grad = False
        # 确保 num_batch 是整数
        num_batch = int(num_samples / batch_size)

        for batch_idx in range(int(num_batch / 2), num_batch):  # 使用 range 确保 batch_idx 是整数
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)  # 确保 end_idx 不超过 num_samples
            sample_ID_batch = np.arange(start_idx, end_idx)

            tau = torch.ones(1) * 0.1
            self.model_parameter[head_predicate_idx]['weight'].requires_grad = True
            optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
            avg_num = 10
            A = self.reparametrization(avg_num, tau)

            # iterate over a batch
            log_likelihood = torch.tensor([0], dtype=torch.float64)
            data_sample = body_time[sample_ID_batch, :]
            t = head_time[sample_ID_batch, :]
            # calculate expectation
            log_p_star, relation_used = self.log_p_star(head_predicate_idx, t, pi, data_sample, A)

            log_avg = log_p_star  # batch_size x num rules
            qzi = F.normalize(torch.exp(log_avg), p=1, dim=1)
            log_likelihood = torch.sum(qzi * log_avg)
            # 通过熵约束增加显著性
            # 损失函数中添加熵约束
            relation_entropy = 0
            for k, v in relation_used.items():
                for predicate_combination in v:
                    if isinstance(self.relation[k][predicate_combination], torch.Tensor):
                        prob = self.relation[k][predicate_combination]
                        relation_entropy += torch.sum(prob * torch.log(prob + 1e-8))
            loss = -log_likelihood +0.1 * relation_entropy
            #closs = -log_likelihood
            loss.backward(retain_graph=True)
            # update weight
            with torch.no_grad():
                grad_weight = self.model_parameter[head_predicate_idx]['weight'].grad
                self.model_parameter[head_predicate_idx]['weight'] -= grad_weight * 0.0002
                self.model_parameter[head_predicate_idx]['weight'] = torch.maximum(self.model_parameter[head_predicate_idx]['weight'], EPSILON)
               
        # 3. get E step
        for batch_idx in np.arange(0, num_batch, 1): # iterate batches
            tau = torch.ones(1)*0.1
            sample_ID_batch = np.arange(batch_idx*batch_size, (batch_idx+1)*batch_size, 1)
            z_p_star = torch.zeros(batch_size,self.num_formula)
            
            avg_num = 10
            A = self.reparametrization(avg_num, tau)
            # iterate over a batch
            log_likelihood = torch.tensor([0], dtype=torch.float64)
            data_sample = body_time[sample_ID_batch,:]
            t = head_time[sample_ID_batch,:]
            # calculate expectation               
            log_p_star, relation_used = self.log_p_star(head_predicate_idx, t, pi, data_sample, A)
            log_avg = log_p_star          # batch_size x num rules
            z_p_star = torch.exp(log_avg)
            qz.append(F.normalize(z_p_star, p=1, dim=1))
            
        qz = torch.cat(qz, dim=0)
        pi = torch.max(F.normalize(torch.sum(qz, dim=0), p=1, dim=0),torch.ones(1)*0.00001)
        pi = F.normalize(pi, p=1, dim=0)
            
        # save result to a txt file
        dict["A"] = A
        print(dict["A"])
        dict["WEIGHT"] = self.weight
        dict["pi"] = pi
        dict["weight"] = self.model_parameter[head_predicate_idx]['weight']
        print("----- A -----")
        print(dict["A"])
        print("----- WEIGHT -----")
        print(dict["WEIGHT"])
        print("----- pi -----")
        print(dict["pi"])
        print('----- weight -----')
        print(dict["weight"])
        return pi.detach()
    
        # 输出所有学习到的规则
    def print_learned_rules(self):
        # 硬编码的 index 到 label 的映射
        index_to_label = {
            0:"Hemoglobin Low",
            1:"WBC High",
            2:"ALT High",
            3:"BUN High",
            4:"Anion gap High",
            5:"Prothrombin time High",
            6:"PTT High",
            7:"Glucoseserum High",
            8:"Albumin Low",
            9:"Chlorideserum Low",
            10:"Glucoseserum Low",
            11:"Sodiumserum Low",
            12:"AST High",
            13:"Lactic Acid Low",
            14:"Arterial CO2 Pressure High",
            15:"PH Arterial Low",
            16:"Heart Rate High",
            17:"Lactic Acid High",
            18:"Heart Rate Low",
            19:"Arterial O2 Saturation Low",
            20:"Chlorideserum High",
            21:"Arterial CO2 Pressure Low",
            22:"Sodiumserum High",
            23:"Potassiumserum Low",
            24:"Hemoglobin High",
            25:"PH Arterial High",
            26:"Total Bilirubin High",
            27:"Prothrombin time Low",
            28:"PTT Low",
            29:"AST Low",
            30:"BUN Low",
            31:"Potassiumserum High",
            32:"Anion gap Low",
            33:"WBC Low",
            34:"ALT Low",
            35:"C Reactive Protein CRP High",
            36:"Albumin High",
            37: "1/2transfer3"
        }

        # 打开文件进行写入
        with open('aki-learned_rules.txt', 'w') as f:
            # 通过 print 将内容同时输出到文件和控制台
            def print_to_file(*args, **kwargs):
                print(*args, **kwargs)  # 打印到控制台
                print(*args, file=f, **kwargs)  # 写入文件
            print_to_file("学习到的所有规则：")
            for rule_id, rule_relations in self.relation.items():
                print_to_file(f"规则 {rule_id}:")
                # 存储当前规则中的最佳谓词组合
                best_body_pair_str = None
                best_relation_value = -1  # 初始化为一个很小的值，确保第一次进入时可以被替换
                for body_predicates, relation in rule_relations.items():
                    body_predicates = eval(body_predicates)  # 转换为实际的谓词索引
                    body_str = " AND ".join([index_to_label[pred] for pred in body_predicates])
                    head_str = index_to_label[self.head_predicate_set[0]]  # 头谓词

                    # 如果 relation 是一个 Tensor，首先转换为列表
                    if isinstance(relation, torch.Tensor):
                        relation = relation.tolist()  # 转换为列表

                    # 找到当前谓词组合中的最大值及其索引
                    max_value = max(relation)
                    max_index = relation.index(max_value)

                    # 跳过“未知”的时间关系（max_index == 3）
                    if max_index == 3:
                        continue

                    # 如果当前谓词组合的权重比之前的最佳值更高，更新最佳谓词组合
                    if max_value > best_relation_value:
                        best_relation_value = max_value

                        # 根据最大值的索引来判断时间关系
                        time_relation = "未知"
                        if max_index == 0:
                            time_relation = "after"
                        elif max_index == 1:
                            time_relation = "equal"
                        elif max_index == 2:
                            time_relation = "before"

                        body_pair = body_predicates[:2]  # 假设只输出体谓词之间的关系
                        best_body_pair_str = f"{index_to_label[body_pair[0]]} {time_relation} {index_to_label[body_pair[1]]}"

                # 输出当前规则的最佳谓词组合（如果存在）
                if best_body_pair_str:
                    print_to_file(best_body_pair_str)

    def organize_rules(self):
        organized_rules = []  # 存储重新整理后的规则
        for rule_id, rule_relations in self.relation.items():
            best_rule = None
            best_score = -float('inf')

            for body_predicates, relation in rule_relations.items():
                # 转换为列表
                if isinstance(relation, torch.Tensor):
                    relation = relation.tolist()
                
                # 找到时间关系的最大值索引及对应的分数
                max_index = relation.index(max(relation))  # 获取时间关系
                max_score = max(relation)

                if max_index == 3:  # 忽略未知关系
                    continue
                
                # 判断当前规则是否比之前的最佳规则更优
                if max_score > best_score:
                    best_score = max_score
                    predicates = eval(body_predicates)
                    best_rule = (predicates[0], predicates[1], max_index)
            
            # 如果找到了最佳规则，则添加到结果中
            if best_rule:
                organized_rules.append(best_rule)

        return organized_rules
    
    def check_rules(self, body_time_sample, rules):
        satisfied_rules = []  # 记录满足的规则
        for idx, rule in enumerate(rules):  # 动态生成规则 ID
            pred1, pred2, time_relation = rule
            time_diff = body_time_sample[pred1] - body_time_sample[pred2]
            
            # 判断时间关系
            if time_relation == 0 and time_diff > self.tolerance:  # after
                satisfied_rules.append((idx, pred1, pred2, time_relation))  # 添加规则 ID
            elif time_relation == 1 and abs(time_diff) <= self.tolerance:  # equal
                satisfied_rules.append((idx, pred1, pred2, time_relation))  # 添加规则 ID
            elif time_relation == 2 and time_diff < -self.tolerance:  # before
                satisfied_rules.append((idx, pred1, pred2, time_relation))  # 添加规则 ID
        
        return satisfied_rules
    
    def compute_intensity(self, body_time_sample, satisfied_rules):
        feature_contributions = []
        weight = self.model_parameter[self.head_predicate_set[0]]['weight']  # 获取规则权重

        for rule in satisfied_rules:
            print(rule)
            rule_id = rule[0]  # 假设规则编号在 rule 的第一个位置
            contribution = weight[rule_id]  # 直接使用规则的权重
            feature_contributions.append(contribution)
        
        # 条件强度 = 基础强度 + 特征贡献之和
        base_intensity = self.model_parameter[self.head_predicate_set[0]]['base']
        total_intensity = base_intensity + sum(feature_contributions)
        return total_intensity

    def predict_head_predicate_time(self, body_time_sample):
        # 整理规则
        rules = self.organize_rules()
        # 判断规则是否符合
        satisfied_rules = self.check_rules(body_time_sample, rules)
        # 计算条件强度
        cur_intensity = self.compute_intensity(body_time_sample, satisfied_rules)
        # 预测时间（条件强度的倒数）
        predicted_time = 1 / cur_intensity.item()
        
        # 找到不为 1.0000e+10 的最大体谓词时间
        valid_times = [time for time in body_time_sample if time < 1.0000e+10]
        if valid_times:
            max_valid_time = max(valid_times).item()
        else:
            max_valid_time = 0  # 如果没有有效时间，默认加0

        adjusted_predicted_time = predicted_time + max_valid_time
        
        return cur_intensity, adjusted_predicted_time,self.model_parameter[self.head_predicate_set[0]]['base'],self.model_parameter[self.head_predicate_set[0]]['weight']
 
if __name__ == "__main__":
    
    print('---------- key tunable parameters ----------')
    print('sigma = 0.1')
    print('tau = 20')
    print('batch_size = 500')
    print('lr = 0.1')
    print('pernalty C = 5')
    print('--------------------')
    # 修改num_samples的数量
    num_samples = 2730
    iter_nums = 1600
    # time_horizon = 5
    # 根据num_samples的数量修改batch_size
    # 使得num_batch是整数
    batch_size = 7
    num_batch = math.ceil(num_samples / batch_size)

    # 加载训练数据
    train_data = np.load('aki_train_data.npy', allow_pickle=True)
    num_train_samples = len(train_data)
    body_time_train = np.zeros((num_train_samples, 37))
    head_time_train = np.zeros((num_train_samples, 1))
    for i, entry in enumerate(train_data):
        #print(len(entry['body_predicates_time']))
        body_time_train[i, :] = entry['body_predicates_time']
        head_time_train[i, :] = entry['head_predicate_time'][0]
    body_time_train = torch.from_numpy(body_time_train)
    head_time_train = torch.from_numpy(head_time_train)

    # 加载测试数据
    test_data = np.load('aki_test_data.npy', allow_pickle=True)
    num_test_samples = len(test_data)
    body_time_test = np.zeros((num_test_samples, 37))
    head_time_test = np.zeros((num_test_samples, 1))
    for i, entry in enumerate(test_data):
        body_time_test[i, :] = entry['body_predicates_time']
        head_time_test[i, :] = entry['head_predicate_time'][0]
    body_time_test = torch.from_numpy(body_time_test)
    head_time_test = torch.from_numpy(head_time_test)

    # 修改头谓词的指针10->37
    head_pred = 37
    logic_model = Logic_Model()
    torch.autograd.set_detect_anomaly(True)
    optimizer = optim.Adam([{'params':logic_model.model_parameter[head_pred]['weight']},{'params': logic_model.weight}], lr=0.1)
    prior = torch.from_numpy(np.array([0.01, 0.99]))

    tau = 20 * torch.ones(1) 
    appearance = {}
    record = []
    record_single = []
    count = 0
    ##### print key model information
    for iter in range(iter_nums):
        prior = logic_model.optimize_EM_single(body_time_train, head_time_train, batch_size, head_pred, optimizer, prior, tau, num_samples)
        if iter == 0:
            prev_weight = torch.ones_like(logic_model.weight.clone())
        else:
            diff = torch.norm(prev_weight-logic_model.weight.clone(), p=1)
            prev_weight = logic_model.weight.clone()

        if count != 0:
            count = 0

        A_m =  logic_model.weight.clone()
        max_ind = torch.sort(A_m,descending=True)[1][:,:logic_model.k]
        print(A_m,max_ind)
        
        valid = np.sort(max_ind, axis=1)  
        for i in range(valid.shape[0]):  
            count = 0
            for j in range(valid.shape[1]):  
                if valid[i, j] >= head_pred:  
                    count += 1
                    valid[i, j] = head_pred + count - 1

        record.append(valid)
        print(record)
        print(appearance)

        # stopping rule 1
        if len(record) >= 4:
            for i in range(-4,-1,1):
                equal = True
                equal = np.array_equal(record[-1],record[i])
                if equal:
                    count += 1
            print(count)
            if count == 3:
                break
            
        record = record[-10:]

    logic_model.print_learned_rules() 
    model_state = {
        'relation': logic_model.relation,  # 保存 relation 字典
        'model_parameter': logic_model.model_parameter  # 保存 model_parameter 字典
    }
    torch.save(model_state, 'aki_model.pt')
    print("模型已保存")

