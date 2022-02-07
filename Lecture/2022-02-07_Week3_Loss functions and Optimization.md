# 3주차: Loss Functions and Optimization

# **Loss Function**

모델의 성능을 나타내는데 사용되는 함수. 손실 함수는 모델 성능의 ‘나쁨’을 나타내는 지표이다. 손실 함수 값이 클 수록 모델의 성능이 낮음을 의미한다. 따라서 손실 함수의 값이 작아지도록 모델을 학습해야 한다.

data set: ![](https://latex.codecogs.com/svg.image?%5C%7B(x_i,~y_i)%5C%7D_%7Bi=1%7D%5EN)  (![](https://latex.codecogs.com/svg.image?x): 입력 데이터, ![](https://latex.codecogs.com/svg.image?y): ![](https://latex.codecogs.com/svg.image?x) 에 대한 레이블).

loss function: ![](https://latex.codecogs.com/svg.image?L_i)

loss: ![](https://latex.codecogs.com/svg.image?L&space;=&space;\frac{1}{N}\sum_{i=1}^N&space;L_i(f(x_i,&space;~W),&space;~y_i))   (![](https://latex.codecogs.com/svg.image?W): 파라미터, ![](https://latex.codecogs.com/svg.image?f): prediction function)

![](https://latex.codecogs.com/svg.image?x) 와 ![](https://latex.codecogs.com/svg.image?W) 를 통해 얻은 예측값과, 실제 정답인 ![](https://latex.codecogs.com/svg.image?y) 를 가지고서 손실값을 구한다. 이를 모든 데이터에 대해 반복한 후 평균을 낸 것이 loss 이며, 모델의 성능을 나타내는 지표다.

### 1. **Multi-Class SVM Loss**

![](https://latex.codecogs.com/svg.image?(x_i,%20~y_i)) 에 대해(![](https://latex.codecogs.com/svg.image?x) 는 사진 데이터, ![](https://latex.codecogs.com/svg.image?y) 는 정수(레이블)), prediction function 을 통해 얻은 예측값을 ![](https://latex.codecogs.com/svg.image?s) 라고 하자. 이때 ![](https://latex.codecogs.com/svg.image?s) 는 각 class 에 대한 예측값을 가지고 있는 vector 이다. ![](https://latex.codecogs.com/svg.image?s_j) 는 ![](https://latex.codecogs.com/svg.image?j) 번째 class 에 대한 점수고, 특히 ![](https://latex.codecogs.com/svg.image?s_{y_i}) 는 정답 class 에 대한 점수다. 이때, multi-class SVM loss function 은 다음과 같다. 

![](https://latex.codecogs.com/svg.image?L_i%20=%20%5Csum_%7Bj%20%5Cne%20y_i%7D%20max(0,%20~s_j-s_%7By_i%7D%20&plus;%201))

만약 정답 class 에 대한 예측 값(![](https://latex.codecogs.com/svg.image?s_{y_i}))이 오답 class 에 대한 예측값(![](https://latex.codecogs.com/svg.image?s_j))에 1을 더한 것 보다 더 크면, ![](https://latex.codecogs.com/svg.image?\sum) 안에 있는 term 은 0이 된다. 다시 말해, 정답 class 에 대한 예측을 오답 class 들에 대한 예측에 비해서 더 잘했다면(여기서는 1보다 더 잘했다면), loss 를 0으로 간주한다는 것이다.

![Untitled](https://user-images.githubusercontent.com/56217002/152763829-e0751f77-2f68-4865-b2ce-2fb2f6a8d674.png)

그림 상의 그래프에서 x 축은 정답 클래스의 예측값(![](https://latex.codecogs.com/svg.image?s_{y_i}))이고, y 축은 loss 이다. 그래프의 모양때문에 multi-class SVM loss 를 hinge loss 라고도 한다. 정답 클래스의 예측값이 나머지 예측값(![](https://latex.codecogs.com/svg.image?s_j))들에 비해 1이 크면 loss 는 0이 된다.

Q1. 일종의 margin 값이라고 할 수 있는 1은 어떻게 정한 것인가?

Q2. multi-class SVM loss 의 최대, 최소값는?

Q3. ![](https://latex.codecogs.com/svg.image?W) 값이 작은 값으로 초기화돼서 모든 예측값(![](https://latex.codecogs.com/svg.image?s))이 0이면 loss 는 얼마인가? 

### **Regularization**

loss 가 0이 되도록 하는 파라미터 ![](https://latex.codecogs.com/svg.image?W) 는 unique 하지 않다. train data 로 모델을 학습시키다 보면, loss 값을 최소로 만드는 파라미터가 하나가 아닌 경우가 존재한다. 이 중 가장 좋은 파라미터를 구하기 위해 regularization 을 사용한다. 다음은 기존 loss function 에 regularization term 을 추가한 식이다.

![](https://latex.codecogs.com/svg.image?L%20=%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi=1%7D%5EN%20L_i(f(x_i,%20~W),%20~y_i)%20~&plus;%20~%5Clambda%20R(W)%20)

regularization term(![](https://latex.codecogs.com/svg.image?%5Clambda%20R(W)%20))이 포함되어 있지 않은 ![](https://latex.codecogs.com/svg.image?L) 은 기존 loss function 와 동일하며, 이는 train data 에 대한 loss 를 구한 것이다. train data 에 대한 loss 를 최소화하는 ![](https://latex.codecogs.com/svg.image?W) 를 찾았다고 하더라도, test data 에 대해선 loss 가 크다면 이는 모델이 train data 에 치우쳐서 학습 했다고 할 수 있다. 이는 오버피팅이 일어난 경우이며, 우리가 모델에게 바라는 바가 아니다.

따라서 test data 에도 좋은 성능을 내도록 하기 위해, 기존 ![](https://latex.codecogs.com/svg.image?L) 에 regularization term 인 ![](https://latex.codecogs.com/svg.image?%5Clambda%20R(W)%20)  을 추가함으로써 모델의 complexity 를 줄인다. 다시 말해, ![](https://latex.codecogs.com/svg.image?%5Clambda%20R(W)%20)  term 을 통해 더 간단한 모델을 선택하게 된다. ![](https://latex.codecogs.com/svg.image?\lambda) 는 하이퍼 파라미터로써, data loss 와 regularization 의 trade-off 를 결정한다. ![](https://latex.codecogs.com/svg.image?\lambda) 값을 조절함으로써, 모델을 train data 에 대해 높은 성능을 보이는 복잡한 모델로 한정할 것인지, 아니면 test data 에 대해서도 좋은 성능을 내도록 하는 간단하고 일반적인 모델을 택할 것인지 결정하게 된다.

regularization 을 통해 더 간단한 모델을 선택하는 방법의 배경에는 Occam’s Razor 가 있다고 한다. Occam’s Razor: “Among competing hypotheses, the simplest is the best”. William of Ockham.

regularization 을 위해 사용되는 것들은 L1, L2, Elastic Net, dropout, batch normalization 등이 있다.

### 2. **Softmax Loss**

![](https://latex.codecogs.com/svg.image?(x_i,&space;~y_i))에 대해(![](https://latex.codecogs.com/svg.image?x) 는 사진 데이터, ![](https://latex.codecogs.com/svg.image?y) 는 정수(레이블)), prediction function 을 통해 얻은 예측값을 ![](https://latex.codecogs.com/svg.image?s) 라고 하자. 이때 ![](https://latex.codecogs.com/svg.image?s) 는 각 class 에 대한 예측값을 가지고 있는 vector 이다. 여기까진 multi-class SVM loss 와 동일하나, softmax loss 는 ![](https://latex.codecogs.com/svg.image?s) 에 대해 softmax 함수를 적용한다. ![](https://latex.codecogs.com/svg.image?s) 에 대해 softmax 함수를 적용하면, ![](https://latex.codecogs.com/svg.image?s) 의 각 원소들은 각 class 에 대한 점수가 아닌, 각 class 에 대한 일종의 확률이 된다.

 ![](https://latex.codecogs.com/svg.image?P(Y=k|X=x_i)&space;=&space;\frac&space;{e^{S_k}}{\sum_j&space;e^{S_j}})    where  ![](https://latex.codecogs.com/svg.image?s&space;=&space;f(x_i,&space;W))

softmax 함수를 적용하면,  ![](https://latex.codecogs.com/svg.image?s)의 ![](https://latex.codecogs.com/svg.image?k) 번째 원소는 데이터가 ![](https://latex.codecogs.com/svg.image?k) 번째 class 에 속할 확률을 의미하게 된다. 그러면, 데이터가 정답 class 에 속할 확률은 ![](https://latex.codecogs.com/svg.image?P(Y=y_i|X=x_i)) 이고, softmax loss 는 다음과 같다. 

![](https://latex.codecogs.com/svg.image?L_i=-logP(Y=y_i|X=x_i)&space;=&space;\frac&space;{e^{S_{y_i}}}{\sum_j&space;e^{S_j}})

확률에 log 를 취함으로써, 정답 class 에 속할 확률을 100% 라고 예측했을때 loss 가 0이 되도록 한다. 그리고 마이너스를 곱한 이유는 loss 의 범위를 0보다 크게 하기 위해서이다.

Q1. softmax loss 의 최대, 최소값는?

Q2. ![](https://latex.codecogs.com/svg.image?W) 값이 작은 값으로 초기화돼서 모든 예측값(![](https://latex.codecogs.com/svg.image?s))이 0이면 loss 는 얼마인가?

SVM loss 는 정답 class 에 대한 점수가 오답 class 에 대한 점수보다 어느 정도 높으면 0이 된다. 반면 softmax loss 는 0이 되려면 정답 class 에 대한 점수가 무한대가 되어야 한다. 즉, SVM loss 는 특정 margin 을 넘으면 학습을 그만두고, softmax loss 는 loss 가 0에 수렴할 때까지 모델을 향상시킨다. 

# **Optimization**

모델의 성능이 좋아지게 만드는 적절한 파라미터를 찾는 과정. 모델의 성능을 측정하기 위해 loss function 을 사용하므로, loss 를 줄이는 방향으로 파라미터를 갱신해 나가야 한다. 적절한 방향으로 파라미터를 갱신하기 위해 사용되는 지표로써, gradient(![](https://latex.codecogs.com/svg.image?\nabla)) 를 사용한다.

gradient 를 사용하는 이유는, gradient 는 각 장소에서 함수값이 가장 증가하는 방향을 가리키는 vector 이기 때문이다. 즉, gradient 의 반대 방향은 함수값이 가장 감소하는 방향을 가리킨다고 볼 수 있다(???). 따라서 loss function 의 값을 줄이는 척도로 gradient 를 사용한다.

**gradient 를 구하는 두 가지 방법**

1. numerical gradient

![](https://latex.codecogs.com/svg.image?\frac{df(x)}{dx}&space;=&space;\lim_{h&space;\to&space;0}&space;\frac{f(x&plus;h)-f(x)}{h})

아주 작은 값에 대해 함수 값의 차이를 구함으로써 기울기를 구하는 방법이다. finite precision 때문에 근사치를 구하는 것이며, loss function 인 ![](https://latex.codecogs.com/svg.image?f) 가 아주 복잡할 경우 계산이 오래걸린다.

2. analytic gradient

수식을 전개해서 기울기를 구하는 방법으로, numerical gradient 방식보다 빠르고 정확하다. analytic gradient 를 사용해서 구한 기울기가 정확한지 확인하기 위해 numerical gradient 방식을 사용해서 검증한다.

**Gradient Descent**

gradient 를 사용해서 파라미터를 갱신하는 방식. 

![](https://latex.codecogs.com/svg.image?W&space;=&space;W&space;-&space;\eta&space;\nabla&space;f)

파라미터에서 gradient 에 해당하는 값을 뺀다. ![](https://latex.codecogs.com/svg.image?\eta) 는 파라미터를 한 번에 얼마만큼 갱신할지 나타내는 하이퍼 파라미터이며, 학습률이라고도 한다.

이 방식은 gradient 를 사용해서 파라미터를 갱신하는 가장 기본적인 방식이라 vanilla gradient descent 라고도 한다. 똑같이 gradient 를 사용하지만 조금 더 똑똑하게 갱신하는 방법으로 momentum, adam 과 같은 방법도 존재한다.

**Stochastic Gradient Descent**

파라미터를 갱신하기 위해 모든 데이터에 대한 loss 와 gradient 를 계산하면 상당한 시간이 걸린다. 따라서 실제로는 전체 데이터 중 일부인 mini batch 데이터에 대해 gradient 를 구하여 파라미터를 갱신하는 방식을 사용하며, 이를 stochastic gradient descent 라고 한다.

mini batch 데이터로부터 구한 loss 와 gradient 를 전체 데이터의 loss 와 gradient 의 추정치라고 보는 것이다. 

# **Image Feature**

딥러닝 방식이 유행하기 전에 image classification 을 위해 널리 사용되었던 방식은 image feature 를 사용하는 것이였다. 이미지에 대해 여러 feature representation 을 얻고, 이를 사용해서 linear classifier 를 통해 분류를 하는 것이 image classification pipeline 이였다. image feature 를 사용해서 데이터를 linear classifier 에 돌릴 수 있도록 linearly separable 하게 만드는 것이 핵심이다.

feature representation 의 예: color histogram, histogram of oriented gradients, bag of words
