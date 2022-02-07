# Lecture 04 - Introduction to Neural Networks

## Back propagation
저번 강의에서 Numerical gradient의 단점과 Analytic gradient의 장점에 대해서 살펴보았는데, Analytic gradient를 구하기 위해 Computational graph를 사용할 수 있다.

**Computational Graph**

![1](https://user-images.githubusercontent.com/79515820/152777468-84f65d10-e431-4642-9f93-c38a92c8f070.png)

위 그래프는 간단한 함수 ![](https://latex.codecogs.com/svg.image?f(x,y,z)%20=%20(x%20&plus;%20y)z) 를 Computational Graph로 나타낸 것이다. 각 연산단계는 노드로 표현된다.

Back propagation은 gradient를 얻기 위해서 Computational Graph 내부의 모든 변수에 대해 Chain rule를 재귀적으로 사용하는 것이다.

우리는 위 Computational Graph에서 ![](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D,%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20y%7D,%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20z%7D) 를 얻기 원한다. 이 때 각각의 gradient를 바로 구하는 것이 아니라, ![](https://latex.codecogs.com/svg.image?q%20=%20x%20&plus;%20y)로 중간 변수를 두고  ![](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%20=%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20q%7D%20%5Cfrac%7B%5Cpartial%20q%7D%7B%5Cpartial%20x%7D)와 같은 Chain rule을 이용해서 구할 수 있다. 이 예시에서는 ![](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D)를 바로 구하는 것이 비교적 간단하지만, 수식이 복잡해지고 단계가 매우 많아지면 gradient를 바로 구하는 것보다 Chain rule을 끝에서부터 재귀적으로 적용하는 편이 훨씬 간단해진다.

위 과정을 좀 더 일반화 해보면  ![](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x%7D%20=%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20q%7D%20%5Cfrac%7B%5Cpartial%20q%7D%7B%5Cpartial%20x%7D) 중 ![](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20q%7D)는 뒤의 노드에서 해당 노드로 들어오는 upstream gradient에 해당하며, ![](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20q%7D%7B%5Cpartial%20x%7D)는 해당 노드에서의 local gradient에 해당한다. upstream gradient의 경우 끝에서부터 계산되어 오기에 이미 계산되어 있는 상태이며, local gradient는 비교적 간단하게 계산할 수 있다. 우리는 끝에서 받아온 upstream gradient와 현재 노드에서의 local gradient를 곱하기만 하면 되므로 연결된 노드 외에 다른 값들을 신경쓸 필요가 없어진다. 이렇게 local gradient를 반복해서 구하는 과정을 통하여 최종적인 gradient를 구할 수 있게 된다.

![2](https://user-images.githubusercontent.com/79515820/152781812-041eb285-e10f-4149-bc7a-e6183ef3d00d.png)

위 그림에서와 같이 복잡한 함수라고 할지라도 Computational Graph를 우선적으로 그리고 끝에서부터 back propagation을 하면 최종적인 gradient를 구할 수 있다. 이 때 노드가 너무 많다면 파란색 네모를 친 부분과 같이 여러 노드를 하나의 그룹으로 묶을 수 있다. 이 경우 역시 그 그룹을 하나의 노드로 보아 그 노드에 대한 local gradient만 구해주면 된다. Graph가 커지더라도 local gradient를 간단하게 계산할 것인지, 아니면 local gradient가 좀 더 복잡해지더라도 노드들을 묶어서 Graph를 좀 더 단순화할 것인지 trade-off가 있으므로 알맞게 결정하면 된다.

![3](https://user-images.githubusercontent.com/79515820/152782471-895be5bc-e46a-47af-94d9-f04bf0978f7f.png)

위 그림에서 초록색 글씨는 각 변수/중간변수들의 값, 빨간색 글씨는 출력값에 대한 각 변수/중간 변수들의 gradient값들이다.
이 그림에서 볼 수 있듯이 +는 gradient값을 앞에 있는 노드들에게 그대로 복사해서 전달해 주는 역할(gradient distributor)을 하며, max는 더 큰 값의 노드에게만 gradient를 주고, 작은 값에는 0을 주는 역할(gradient router)을 한다. *에서 local gradient는 다른 변수의 값(예를 들어 a\*b에서 a는 b)에 해당하므로 이 경우 gradient switcher의 역할을 한다.

![4](https://user-images.githubusercontent.com/79515820/152783140-f1ebc447-6679-44c0-adcd-40dfdc2573f0.png)

위 그림에서와 같이 한 노드에서 branch되는 경우가 있을 수도 있는데, 이 경우 upstream gradient는 각 노드에서 들어오는 upstream gradient의 합이 된다.

![5](https://user-images.githubusercontent.com/79515820/152783495-9772f275-4cdb-4a02-a566-cba6adb15dad.png)

앞에서까지의 back propagation에서 스칼라가 아닌 벡터로 확장할 수도 있는데, 이 때의 local gradient는 Jacobian Matrix(각 행이 입력값에 대한 출력값의 편미분이 되는 행렬)가 된다.
다만 위 사진에서와 같이 input vector와 output vector의 dimension이 4096이고 minibatch가 100이면(100개의 input을 동시에 입력받아 작업) Jacobian은 409600 x 409600으로 너무 커져 버리게 된다. 하지만 입력의 각 요소는 출력의 해당 요소에만 영향을 준다는 점을 이용하면 그 거대한 Jacobian을 모두 계산할 필요가 없다.

![6](https://user-images.githubusercontent.com/79515820/152784566-8d54dc26-809e-4a21-92d3-06dc990f9701.png)

x가 2차원 벡터, W가 2*2인 예시를 보면, 우선 W와 x의 내적한 것을 중간변수 q로 둘 수 있고 f(x, W)를 q의 L2 norm으로 볼 수 있다. 이 경우 f를 qi로 미분한 것은 2qi가 되므로 * 노드에서의 gradient는 [0.44 0.52]T 가 된다.

![7](https://user-images.githubusercontent.com/79515820/152785352-4ffab7c3-b3cb-438a-8b72-746d2f65c2ac.png)

마찬가지로 local gradient와 upstream gradient를 곱해줘서 f를 W와 x에 대해 미분한 gradient도 구할 수 있다. 위 그림은 W의 경우를 나타낸 것이다.  여기서 ![](https://latex.codecogs.com/svg.image?1_%7Bk=i%7D%5E%7B%7D)는 k가 i일 때는 1, 아니면 0을 나타내는데, 이는 q의 n번째 행은 W의 n번째 행에만 영향이 있기 때문에 이러한 항이 나타난 것이다.


## Neural Network
![8](https://user-images.githubusercontent.com/79515820/152786024-32de38d4-1d4d-4776-9199-61a6d3eefe67.png)

지금까지 배운 score function이 ![](https://latex.codecogs.com/svg.image?f%20=%20Wx)와 같은 Linear 형태였다면, Neural Network에서는 ![](https://latex.codecogs.com/svg.image?f%20=%20W_%7B2%7D%5E%7B%7Dmax(0,W_%7B1%7Dx))와 같이 선형 함수에 비선형 함수를 취하고, 그것을 다시 선형 함수에 넣는 식으로 여러 레이어로 구성된다.
이 때 중간에 비선형 레이어가 반드시 필요한데, 그렇지 않으면 선형함수에 여러번 넣어봤자 결국 하나의 선형함수의 결과로 나올 수 밖에 없기 때문이다.
기본적으로 신경망은 비선형의 복잡한 함수를 만들기 위해 간단한 함수들을 계층적으로 여러개 쌓아올린 함수들의 집합을 말한다.
![9](https://user-images.githubusercontent.com/79515820/152786821-cd4d63ef-2b9c-4467-8a17-4c248c0f15ce.png)

이전 강의의 image classification 부분의 템플릿을 다시 가져와 보면, 위 그림에서 h는 W1에서 가지고 있는 여러 템플릿에 대한 스코어 값이다. W2는 템플릿 모두에 가중치를 부여하고 모든 중간점수를 더해 클래스에 대한 최종점수를 얻도록 한다.
기존에는 하나의 car에 대해서 하나의 템플릿만을 가져 여러 다른 종류의 차를 찾기에 어려움이 있었는데(예를 들어 템플릿은 빨간차인데 파란차도 인식하기를 원하는 경우) Multiple Neural network를 통해 이를 해결할 수 있게 된다.

![10](https://user-images.githubusercontent.com/79515820/152787598-4f8a9b17-216a-4548-a082-3c52c5a15e88.png)

Neural Network의 각 Computational node는 실제 뉴런과 비슷한 방식으로 동작한다고 볼 수 있다. 입력 신호 x는 가중치 W와 결합하여 합쳐진 후 activation function이 적용되어 output으로 다음 뉴런에게 전달된다.

![11](https://user-images.githubusercontent.com/79515820/152787430-8b940f5d-0670-4700-9101-8133b35fb89a.png)

이 때 핵심이 되는게 activation function으로 이러한 비선형 함수가 없다면 여러 뉴런을 거쳐봤자 결국 선형함수 하나를 거친 꼴이 되어버린다. 위와 같은 여러 activation function들이 존재하는데, ReLU가 실제 뉴런과 가장 유사하게 동작한다고 신경과학자들이 말한다고 한다.