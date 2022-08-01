---
layout: post
title:  "DQN(Deep Q-Network) & Cartploe-v1"
summary: DQN 요약  & Cartpole 코드 리뷰
author: yeop-giraffe
date: '2022-08-02'
category: [Coding, RL]
thumbnail: /assets/img/posts/code.jpg
---

# DQN(Deep Q-Network)
___
##  CNN, Experience replay, Target network 세가지 특징으로 구성되어 있다.
<br/>


# Cartpole-vl
## 1. Import library & Parameters
___
```python
import gym
import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Hyberparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
```
- torch : pytorch 사용
- collections : deque 사용
- buffer_limit : 데이터 저장 limit
- batch_size : 연산 한번에 들어가는 데이터의 크기, 1 batch_size에 해당하는 데이터 셋을 mini_batch라고 한다.
<br/><br/>

## 2. ReplayBuffer()
*지금까지 데이터를 저장(buffer)했다가 랜덤하게 추출해서 연관성이 적은 데이터를 사용*
```python
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)
```

- deque : 입구와 출구가 모두 열려있는 데이터 구조, 최근 maxlen만큼의 데이터를 저장, 넘치면 오래된 데이터부터 삭제
- put : transition을 buffer에 추가
- mini_batch : buffer에서 n개 만큼의 데이터를 랜덤으로 추출, *random.sample(sequence, k)*
- transition : 이행 데이터 (s,a,r,s')
- torch.tensor :
  * tensor : array, matrix와 유사한 자료구조, 3차원 이상 의미
  * tensor 반환
- size : buffer 크기 변환
<br/><br/>

## 3. Qnet(nn.Module)
*여러 개의 레이어로 구성된 뉴럴 네트워크, pytorch에서 레이어를 구성할 때는 nn.Module을 상속받아야 된다.*
```python
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random() 
        if coin < epsilon:    
            return random.randint(0,1)
        else : 
            return out.argmax().item()
```
- super().__ init__() : 기반 클래스를 초기화해서 기반 클래스의 속성을 subclass가 받아오도록 한다. [출처] [[Pytorch] nn.Module & super().__init__()](https://daebaq27.tistory.com/60)
- nn.Linear(in_features, out_features) : fully connected layers(fc1, fc2, fc3)를 선언
- forward : 
  * relu : Rectified Linear Unit의 약자로 0보다 작으면 0, 0보다 크면 입력값 그대로를 반환
  * 데이터가 layers를 통과, 4차원->128차원->2차원
- sample_action : 랜덤 값 반환
</br></br>

## 4. 

```python
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)                                
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) 
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```