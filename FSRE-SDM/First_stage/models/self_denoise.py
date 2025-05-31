import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from . import gnn_iclr
import pdb
import copy
from transformers.models.bert.modeling_bert import ACT2FN
import math
import random
from .Model import MetaModule
from .L2P import *
from torch.distributions import Dirichlet

def pred(query, Poto, dot=True, sqrt=False):
    if dot:
        return torch.matmul(query, Poto.transpose(-1, -2))
    else:
        if(sqrt):
            return -torch.sqrt((torch.pow(query.unsqueeze(2) - Poto.unsqueeze(1), 2)).sum(-1) + 1e-6)
        else:
            return -(torch.pow(query.unsqueeze(2) - Poto.unsqueeze(1), 2)).sum(-1)

class ATLoss(nn.Module):
    def __init__(self, K, theta=-100.0):
        super().__init__()
        self.theta = theta
        self.K = K
    def forward(self, logits, labels):
        '''
        :param logits: [B,N,L]
        :param labels: [B,N,L]
        :return:
        '''
        B, N, L = labels.size()
        values, indices = logits.topk(self.K + 1, dim=-1, largest=True, sorted=True)
        theta = values[:, :, -1:].detach()  # [B,N,1]
        logits = torch.cat((theta, logits), dim=-1).view(B * N, -1)
        # logits = torch.cat(((1.0 - label_0) * self.theta, logits), dim=-1).view(B * N, -1)
        label_0 = torch.zeros_like(labels[:, :, :1])
        labels = torch.cat((label_0, labels), dim=-1).view(B * N, -1)

        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        B, N, L = logits.size()
        label_0 = torch.zeros_like(labels[:,:,:1])
        logits = torch.cat(((1.0 - label_0) * self.theta, logits), dim=-1).view(B * N, -1)

        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)

        output = output[:, 1:].view(B, N, -1)
        return output




class GAT(nn.Module):
    def __init__(self, config, num_attention_heads=4):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        self.query = nn.Sequential(
            nn.Linear(config.hidden_size * 2, self.hidden_size),
            # nn.LeakyReLU(0.1)
            # nn.ReLU(inplace=True),
        )
        self.key = nn.Sequential(
            nn.Linear(config.hidden_size * 2, self.hidden_size),
            # nn.LeakyReLU(0.1)
            # nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(
            nn.Linear(config.hidden_size * 2, self.hidden_size),
            # nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.out_put = nn.Linear(config.hidden_size, self.hidden_size)

    def transpose_for_scores(self, x):
        attention_head_size = x.size(-1) // self.num_attention_heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, samples):
        '''
        :param samples: # [B,L,d]
        :return:
        '''
        # [B,h,L,d]
        query = self.transpose_for_scores(self.query(samples))
        key = self.transpose_for_scores(self.key(samples))
        value = self.transpose_for_scores(self.value(samples))

        # [B,h,L,L]
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # [B,h,L,d]
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        samples = self.out_put(context_layer)
        return samples

class Denoise_module(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.graph_propa = GAT(config)
        self.Label_Propa = Poisson_learning(config)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.Train = True
        self.label_g = None
        self.label_n = None

    def to_one_hot(self, Predict, N):
        j = 0
        Predict_tmp = Predict.detach().clone().transpose(-1, -2)
        while (True):
            values, indices = Predict_tmp.topk(N, dim=-1, largest=True, sorted=True)
            Mask2 = (Predict_tmp >= values[:, :, -1:]).float()
            if ((Mask2.sum(-2) < 2).all()):
                break


            Predict_tmp1 = Predict_tmp - (1.0 - Mask2) * 1000000.0
            value, _ = torch.max(Predict_tmp1, dim=-2)
            value = value.unsqueeze(-2)
            Mask3 = (Predict_tmp1 >= value).float()

            Mask4 = Mask3 * Mask2
            Mask5 = (Mask4.sum(-2) > 0.5).float().unsqueeze(-2)
            Predict_tmp2 = Predict_tmp - Mask5 * 1000000.0

            Mask6 = (Mask4.sum(-2) > 1.5).float().unsqueeze(-2)

            Mask4 = Mask4 * (1.0 - Mask6)
            Predict_tmp = Mask4 * Predict_tmp + (1.0 - Mask4) * Predict_tmp2

            j += 1
            if (j > 10):
                break

        return Mask2.transpose(-1, -2)


    def Kmean(self, sample, relation, Label, graph):
        '''
        :param sample: [B,L,D]
        :param label: [B,L,N]
        :return:
        '''
        B, L, N = Label.size()
        Logits = []
        i = 0
        label = Label
        for i in range(4):
            if (self.Train):
                mask_l = ((label * self.label_g).sum(-2) < 2).float().unsqueeze(-2)
                if (mask_l.sum() > 0):
                    label_n = self.label_n * mask_l
                    mask_n = (label_n.sum(-1) > 0.5).float().unsqueeze(-1)
                    label = (1 - mask_n) * label + mask_n * label_n

            Mask = torch.matmul(label, label.transpose(-1, -2))  # [B,L,L]
            dis_mask = graph * Mask + (1.0 - Mask) * -1000000.0
            sample_w = torch.max(dis_mask,dim=-1)[0]

            label_tmp = label.transpose(-1, -2)  # [B,N,L]
            label_tmp = label_tmp * sample_w.unsqueeze(-2) + (1.0 - label_tmp) * -1000000.0 # [B,N,L]
            label_tmp1 = F.softmax(label_tmp, dim=-1)  # [B,N,L]

            Center = ((N + 1) / (N + 2)) * torch.matmul(label_tmp1, sample) + (1 / (N + 2)) * relation  # [B,N,D]

            Predict = pred(sample, Center, dot=False)  # [B,L,N]
            Logits.append(Predict)

            label = self.to_one_hot(Predict, N+1)

        Logits = torch.stack(Logits, dim=0).mean(0)
        return label, Center, label_tmp1, Logits, Predict

    def adj_matrix(self, sample):
        B, L, D = sample.size()
        Graph = pred(sample, sample, dot=False, sqrt=True)  # [B,L,L]
        diag = torch.eye(L).unsqueeze(0).expand(Graph.size()).to(Graph)
        diag = diag * -1000000.0
        Graph = Graph + diag
        return Graph

    def Center_gate(self, samples, label, d=0):
        socore1 = label.detach().transpose(-1, -2)
        socore1 = socore1 / socore1.sum(-1).unsqueeze(-1)
        Center = torch.matmul(socore1, samples)

        socore2 = label.detach()
        socore2 = socore2 / socore2.sum(-1).unsqueeze(-1)
        samples_tmp = torch.matmul(socore2, Center)

        g = F.sigmoid(self.gate[d](torch.cat((samples, samples_tmp), dim=-1)))
        samples_out = g * samples + (1 - g) * samples_tmp

        return samples_out

    def forward(self, samples, relation, label):
        '''
        :param samples: [B,L,d]
        :param label [B,L,N]
        :return:
        '''
        B, L, N = label.size()
        samples_p = torch.cat((samples, relation), dim=-2)
        samples_d = torch.cat((self.graph_propa(samples_p), samples_p), dim=-1)
        relation_d = samples_d[:, L:]
        samples_d = samples_d[:, :L]

        graph = self.adj_matrix(samples_d)
        label_new, Center, Map, logit, Predict = self.Kmean(samples_d, relation_d, label.clone(), graph)

        label_new1 = self.Label_Propa(samples_d, label_new.clone())

        return Predict, logit, samples_d, relation_d, label_new1

class Poisson_learning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dropout_noise = nn.Dropout(p=0.3)

        self.Pro_ject = nn.Linear(config.hidden_size * 4, config.hidden_size)

    def label_forward(self, label, edge, T=6):
            '''
            :param label: [b,L,N]
            :param edge: [b,L,L]
            :param N:
            :return:
            '''
            b, l, _ = edge.size()
            B = label.clone()  #[b,L,N]

            I = torch.eye(l).unsqueeze(0).expand(b, l, l).to(edge)
            D = edge + 1e-10 * I

            D = torch.sum(D, -1) ** -1
            D_temp =[]
            for i in range(b):
                D_temp.append(torch.diag(D[i]))

            D = torch.stack(D_temp,dim=0)  #[b,l,l]


            P = torch.matmul(D, edge.transpose(-1, -2))
            Db = torch.matmul(D, B)

            ut = torch.zeros_like(B)
            t = 0
            while t < T:
                ut = torch.matmul(P, ut) + Db
                t = t+1
                # ut = self.dropout(ut)
                if t == 3 or t == 7:
                    # ut = ut + predict
                    ut = self.dropout(ut)

            socore = torch.matmul(ut, ut.transpose(-1, -2))
            return socore, ut

    def forward(self, samples, label_one_hot):
        samples = self.Pro_ject(samples)

        Graph = torch.matmul(samples, samples.transpose(-1, -2))
        b, L, _ = Graph.size()
        Graph_no_diag = torch.eye(L).unsqueeze(0).expand(b, L, L).to(Graph)
        Graph_no_diag = Graph_no_diag * -100000.0
        Graph_no_diag = Graph_no_diag + Graph
        Graph_no_diag = nn.Softmax(dim=-1)(Graph_no_diag)


        predict_sup = label_one_hot
        Total = (predict_sup.sum(-1) > 0.5).float().sum(-1).unsqueeze(-1)
        predict_sup_avg = (predict_sup.sum(-2) / Total).unsqueeze(-2)
        Mask = (predict_sup.sum(-1) > 0.5).float().unsqueeze(-1)
        Label_origin = (predict_sup - predict_sup_avg) * Mask


        scores, label = self.label_forward(Label_origin, Graph_no_diag)
        Label = nn.Softmax(dim=-1)(label / 0.1)

        return label

class Proto_Net(MetaModule):
    def __init__(self, config):
        super(Proto_Net, self).__init__()
        self.Pro_ject = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            # nn.ReLU(inplace=True),
            nn.Tanh(),
            nn.Dropout(config.attention_probs_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, Sample):
        return self.Pro_ject(Sample)

class Self_Denoise(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, K, hidden_size=768):
        '''
        N: Num of classes
        '''
        self.config = sentence_encoder.config
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)

        self.GAT_label = Denoise_module(self.config)
        self.Poto_layer = Proto_Net(self.config)
        self.ATLoss = ATLoss(K)
        self.dropout = nn.Dropout(self.config.attention_probs_dropout_prob)
        self.dirichlet = Dirichlet(torch.FloatTensor([1.0] * 6))


    def forward(self, sample_sup, sample_que, relation, label, N, K, Q, Train=True):
        relation = self.sentence_encoder(relation, example=False)  # [b*r_n,d]
        sup_set = self.sentence_encoder(sample_sup)  # [b*r_n,d]
        que_set = self.sentence_encoder(sample_que)  # [b*r_n,d]

        D = que_set.size(-1)
        que_set = que_set.view(-1, N * Q, D)
        sup_set = sup_set.view(-1, N * K, D)
        relation = relation.view(-1, N, D)

        B = que_set.size(0)
        label_one_hot = label.view(B, -1, N).float()
        label_index = label_one_hot.max(-1)[-1]

        sup_label = label_one_hot[:, :N * K].clone().view(B, N, K, N)  # [B,N,K,N]
        for i in range(3):
            s_i = sup_label[:, :, i] #[B,N,N]
            s_i_noise = torch.cat((s_i[:, (i + 1):], s_i[:, :(i + 1)]), dim=-2)
            sup_label[:, :, i] = s_i_noise
        label_one_hot_noise = torch.cat((sup_label.view(B, N * K, N), label_one_hot[:, N * K:]), dim=-2)


        Poto_mask = label_one_hot.transpose(-1, -2)
        Poto_mask = Poto_mask / Poto_mask.sum(-1).unsqueeze(-1)
        samples = torch.cat((sup_set, relation), dim=-2)
        Poto = torch.matmul(Poto_mask, samples)
        Predict = pred(que_set, Poto, dot=False)
        Loss = self.loss(Predict, label_index[:, N * K:].long())


        Samples = torch.cat((sup_set.view(B, N, K, D), relation.view(B, N, 1, D)),dim=-2)
        sup_set_add = []
        for i in range(K):
            alpha = self.dirichlet.sample_n(B * N).view(B, N, -1, 1).to(Samples)
            sample = (alpha * Samples).sum(-2)
            sup_set_add.append(sample)
        sup_set_add = torch.stack(sup_set_add,dim=-2)

        sup_set_add_PNM = self.Poto_layer.Pro_ject(sup_set_add)
        que_set_LCM = self.Poto_layer.Pro_ject(que_set)

        Predict1 = pred(que_set_LCM, sup_set_add_PNM.mean(-2), dot=False)
        Loss1 = 0.1 * self.loss(Predict1, label_index[:, N * K:].long())

        sup_set_add1 = []
        for i in range(K):
            alpha = self.dirichlet.sample_n(B * N).view(B, N, -1, 1).to(Samples)
            sample = (alpha * Samples).sum(-2)
            sup_set_add1.append(sample)
        sup_set_add1 = torch.stack(sup_set_add1, dim=-2).view(B, N * K, D)

        samples_in = torch.cat((sup_set, que_set, sup_set_add1), dim=-2)
        samples_LCM = self.GAT_label.graph_propa(samples_in)
        sup_que_set = samples_LCM[:, :N * (K + 1)]
        Poto2 = samples_LCM[:, N * (K + 1):].contiguous().view(B, N, K, -1).mean(-2)

        Predict2 = pred(sup_que_set, Poto2, dot=False)
        Loss2 = 0.1 * self.ATLoss(Predict2.transpose(-1, -2), label_one_hot.transpose(-1, -2))


        Predict2 = Predict2[:, N * K:]
        predict_new = [Predict.max(-1)[-1].flatten(), Predict1.max(-1)[-1].flatten(), Predict2.max(-1)[-1].flatten()]
        label0 = label_index[:, N * K:].flatten()

        Loss_tmp = torch.tensor([Loss, Loss1, Loss2])
        if(torch.isnan(Loss_tmp).any() or torch.isinf(Loss_tmp).any()):
            pdb.set_trace()

        return predict_new, label0, Loss, Loss1+Loss2