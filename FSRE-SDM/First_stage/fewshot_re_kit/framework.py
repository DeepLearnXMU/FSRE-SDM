import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import sentence_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import pdb
from torch.cuda.amp import autocast as autocast, GradScaler


def output_model_put(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
        print(parameters.grad)


def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder)
        self.cost = nn.CrossEntropyLoss()
        self.cost_b = nn.BCEWithLogitsLoss()
    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label, Bi = False):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        loss = 0.0
        if(Bi):
            loss = self.cost_b(logits,label)
        else:
            N = logits.size(-1)
            loss = self.cost(logits.contiguous().view(-1, N), label.contiguous().view(-1))

        return loss

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

class FewShotREFramework:
    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              na_rate=0,
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-4,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              pair=False,
              adv_dis_lr=1e-1,
              adv_enc_lr=1e-1,
              use_sgd_for_bert=False):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        print("Start training...")

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['sentence_encoder']
        component1 = ['GAT_label', 'Poto_layer','GCN']
        parameters = [
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay) and any(nd in n for nd in component)],
             'weight_decay': weight_decay,
             'lr': learning_rate
             },
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay) and any(nd in n for nd in component)],
             'weight_decay': 0.0,
             'lr': learning_rate
             },
            {'params': [p for n, p in parameters_to_optimize
                        if not any(nd in n for nd in no_decay) and any(nd in n for nd in component1)],
             'weight_decay': weight_decay,
             'lr': learning_rate * 5
             },
            {'params': [p for n, p in parameters_to_optimize
                        if any(nd in n for nd in no_decay) and any(nd in n for nd in component1)],
             'weight_decay': 0.0,
             'lr': learning_rate * 5
             }
        ]

        if use_sgd_for_bert:
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(parameters, lr=learning_rate, correct_bias=False)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        # Training
        best_acc = 0.80
        iter_loss, iter_loss1 = 0.0, 0.0
        iter_loss_dis = 0.0
        iter_right = [0.0, 0.0, 0.0]
        iter_right_dis = 0.0
        iter_sample = 0.0
        time_start = time.time()
        with torch.autograd.set_detect_anomaly(True):
            for it in range(start_iter, start_iter + train_iter):
                sample_sup, sample_que, relation, label = next(self.train_data_loader)
                if torch.cuda.is_available():
                    for k in sample_sup:
                        sample_sup[k] = sample_sup[k].cuda()
                        sample_que[k] = sample_que[k].cuda()
                        relation[k] = relation[k].cuda()
                    label = label.cuda()

                model.train()
                pred, label, Loss, Loss1 = model(sample_sup, sample_que, relation, label, N_for_train, K, Q + na_rate * Q)

                loss = Loss + Loss1
                loss /= float(grad_iter)
                right = [model.accuracy(pred[0], label), model.accuracy(pred[1], label), model.accuracy(pred[2], label)]

                loss.backward()
                if it % grad_iter == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                iter_loss += self.item(Loss.data)
                iter_loss1 += self.item(Loss1.data)
                iter_right[0] += self.item(right[0].data)
                iter_right[1] += self.item(right[1].data)
                iter_right[2] += self.item(right[2].data)

                iter_sample += 1
                # val_step = 100
                if (it + 1) % val_step == 0:
                    acc = self.eval(model, B, N_for_eval, K, Q, val_iter, na_rate=na_rate, pair=pair)
                    model.train()
                    if acc > best_acc:
                        # print('Best checkpoint')
                        torch.save({'state_dict': model.state_dict()}, save_ckpt)
                        best_acc = acc
                    else:
                        print(acc)
                    iter_loss = 0.
                    iter_loss1 = 0.
                    iter_loss_dis = 0.
                    iter_right = [0.0, 0.0, 0.0]
                    iter_right_dis = 0.
                    iter_sample = 0.
                
        print("\n####################\n")
        print("Finish training " + model_name)

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            na_rate=0,
            pair=False,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''

        model.eval()
        if ckpt is None:
            # print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_right = [0.0, 0.0, 0.0]
        iter_sample = 0.0
        with torch.no_grad():
            for it in range(eval_iter):
                sample_sup, sample_que, relation, label = next(self.val_data_loader)
                if torch.cuda.is_available():
                    for k in sample_sup:
                        sample_sup[k] = sample_sup[k].cuda()
                        sample_que[k] = sample_que[k].cuda()
                        relation[k] = relation[k].cuda()

                    label = label.cuda()

                pred, label, _, _ = model(sample_sup, sample_que, relation, label, N, K, Q + na_rate * Q, Train=False)

                right = [model.accuracy(pred[0], label),model.accuracy(pred[1], label),model.accuracy(pred[2], label)]

                iter_right[0] += self.item(right[0].data)
                iter_right[1] += self.item(right[1].data)
                iter_right[2] += self.item(right[2].data)
                iter_sample += 1
            #     if (it + 1) % 100 == 0:
            #         sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}% ; accuracy: {2:3.2f}% ; accuracy: {3:3.2f}%'.format(
            #         it + 1, 100 * iter_right[0] / iter_sample, 100 * iter_right[1] / iter_sample, 100 * iter_right[2] / iter_sample) + '\r')
            #         sys.stdout.flush()
            # print("")
        return iter_right[0] / iter_sample
