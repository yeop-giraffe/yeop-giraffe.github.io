---
layout: post
title:  "DQN(Deep Q-Network) & Cartploe-v1"
summary: DQN 요약  & Cartpole 코드 리뷰
author: yeop-giraffe
date: '2022-08-02'
category: [Coding, RL]
tags: dqn
thumbnail: /assets/img/posts/code.jpg
---

# DQN(Deep Q-Network)
___
## Q-learning 차이점
- 기존 Q-learning의 경우 테이블에 value를 저장하여 학습을 진행하기 때문에 state와 action의 경우가 많아지면 테이블로 표현하는 것이 불가능해진다. 이러한 한계를 극복하기 위해 딥러닝을 사용한다.
- 2013년 "Playing Atari with Deep Reinforcement Learning"이라는 주제로 DeepMind에서 DQN을 처음 제시했다.
- CNN, Experience replay, Target network 세가지 특징으로 구성되어 있다.
</br>

## 특징점
### 1) CNN
- 이미지 처리에 뛰어난 알고리즘
- CNN의 입력으로 state만을 받으며 출력으로 Q-value를 얻는다. 

### 2) Experience Replay
- experience를 버퍼에 저장을하고 나중에 랜덤하게 추출하여 학습을 업데이트하는데 사용한다.
- Data efficiency 증가 : 하나의 데이터를 여러번 업데이트에 사용 가능
- Sample correlation 감소 :  랜덤으로 샘플을 추출하기 때문에 데이터 사이의 연관 감소
- Data distribution 해결 : on-policy의 경우 현재의 파라미터로 인해 policy와 training data가 변하는 문제가 있는데 Ex replay로 인해 평균화 되기 때문에 문제가 발생하지 않는다.

### 3) Target Network
- 기존 Q-network의 경우, target value는 동일한 시점의 파라미터로 구성되어 있었다. 이는 파라미터가 업데이트됨에 따라 action, target value가 동시에 변화하여 수렴하지 못하는 문제가 발생했다.
- DQN에서는 target value를 기존 Q-network와 동일하게 복제를 하여 main Q-network와 target network로 구성을 한다. 이후 C번의 step동안 target value를 고정시켜서 모델을 업데이트하고 그 이후 새롭게 target value를 설정하여 다시 반복하는 과정을 만든다.
  
</br>
  
  


# Cartpole-v1
___
## 1. Import library & Parameters
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
<br/>

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
<br/>

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
</br>

## 4.train(q, q_target, memory, optimizer) 
*학습하는 과정, Q(quality)는 행동의 보상의 가치라는 뜻으로 Q(s,a)는 특정 state에서 action을 취할 때 그 행동이 갖고 있는 가치를 반환하는 함수를 의미한다.*
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
- memory.sample(batch_size) : batch_size(32)크기의 sample을 추출해서 각 변수에 할당
- torch.gather() : 
- torch.unsqueeze(input, dim) : input(tensor)을 dim(int)크기의 dimension의 텐서로 변환
  
    Example
    >  x = torch.tensor([1, 2, 3, 4])</br>
    > 
    > torch.unsqueeze(x, 0)</br>
    > -> tensor([[ 1,  2,  3,  4]])</br>
    >
    >  torch.unsqueeze(x, 1)</br>
    > -> tensor([[ 1], [ 2], [ 3], [ 4]])
- loss : smooth_l1_loss를 사용해 손실 값 계산
</br>

## 5. main()
  *코드 실행*

```python
def main():
    env = gym.make('CartPole-v1')
    q = Qnet() # Neural network 통과
    q_target = Qnet() # Neural network 통과
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()

```

- q, q_target : Neural network 통과
- load_state_dict(), state_dict() : 각각 불러오기, 저장하기를 의미하며 q_target을 저장, 불러오기 진행
- optimizer :
- env.reset() : 새로운 환경 불러오기
- q.sample.action : ???
- env.step(a) : a라는 행동을 취했을 때 획득한 환경 정보 리턴
- memory.put() & s = s_prime : 메모리에 step의 정보를 입력하고 s를 s'으로 변경
- memory.size()>2000: 메모리 사이즈가 2000개가 넘을 때만 학습 진행, 샘플이 너무 적으면 안됨 

