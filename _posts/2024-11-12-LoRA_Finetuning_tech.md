# LoRA : LLM의 효율적인 파인튜닝 기법
LLM은 여러 모로 이점이 있고 유용하지만, 파인튜닝에 있어서는 여전히 문제가 있었다.

기존의 파인튜닝 방식은 모든 파라미터를 업데이트하는 방식을 이용했으므로 Computational Resource와 메모리를 엄청 잡아먹는다.

`LoRA(Low-Rank Adaptation)`은 이러한 문제를 해소하는 기법이다.
      
      

# 기존 동작 방식과 LoRA의 차이점    

기존 방식 : 
- 무식하게 모든 파라미터를 업데이트시키는 방식      
- $$h = Wx$$ 에서 모든 W를 직접 업데이트

LoRA : 
- 파인튜닝된 모델의 가중치는 고정시키고, `Low-Rank Decomposition행렬`을 통해 가중치 업데이트를 근사하는 방식
- `Low-Rank Decomposition행렬` A, B를 도입하여 가중치의 업데이트를 근사한다.
- $$h = Wx + BAx$$    

즉 LoRA의 핵심은, `Pre-train된 모델의 가중치를 고정시키고`, `Low-Rank Decomposition행렬을 통해 가중치의 업데이트를 근사하는 것` 이다.

# 뭐가 좋을까?

1. 메모리를 효율적으로 사용할 수 있다.
  학습 가능한 파라미터의 수를 대폭 줄여, GPU 메모리 사용량을 크게 감소시킨다.      
  일례로, GPT-3 175B 모델의 경우, 메모리 요구사항을 1.2TB $\to$ 350GB까지 줄일 수 있다고 한다.      
      
2. 학습 속도를 향상시킬 수 있다.
   파라미터 수의 감소를 통해, 학습 속도 또한 개선된다.

3. 유연하다.
  다양한 downstream task에 대해 적은 수의 task-specific한 파라미터들만으로 모델을 조정할 수 있다.

4. 추론 시 지연이 거의 없다.
  학습된 LoRA 가중치를 원본 모델과 병합하여, 추론 시 지연 없이 거의 바로 사용할 수 있다.

# 코드를 통한 비교
간단한 선형 Layer를 통해 LoRA를 쓴것과 안쓴것의 차이를 보자.

1. 일반적인 방식

```python
import torch
import torch.nn as nn

class NormalLinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# 사용 예시
model = NormalLinearLayer(768, 768)
x = torch.randn(1, 768)
output = model(x)
print(output)
```

2. LoRA

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling

class LinearWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora = LoRALayer(in_features, out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

# 사용 예시
model = LinearWithLoRA(768, 768, rank=8)
x = torch.randn(1, 768)
output = model(x)
```


