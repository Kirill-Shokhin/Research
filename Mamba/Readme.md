# Mamba or first Selective SSM
![image (7)](https://github.com/Kirill-Shokhin/Research/assets/46619252/b6209055-8de7-43ae-902e-f116735ef9b2)

## General state space model
### Linear continuous time-invariant SSM

Оригинальный вид непрерывной модели пространства состояний выглядит так:

$$ \boldsymbol{\dot h}(t) = \boldsymbol{Ah}(t)+\boldsymbol{Bx}(t) $$

$$\boldsymbol{y}(t) = \boldsymbol{Ch}(t)+\boldsymbol{Dx}(t) $$

Чтобы взаимодествовать с ней в векторном виде конечной размерности, нужно дискретизовать ее.

### Discretization of linear time-invariant SSM
Умножим первое уравнение на $`e^{-\boldsymbol{A}t}`$:

$$ e^{-\boldsymbol{A}t} \boldsymbol{\dot h}(t) - e^{-\boldsymbol{A}t} \boldsymbol{Ah}(t) = e^{-\boldsymbol{A}t} \boldsymbol{Bx}(t)$$

$$ \frac{d}{dt} (e^{-\boldsymbol{A}t} \boldsymbol{h}(t)) = e^{-\boldsymbol{A}t} \boldsymbol{Bx}(t)$$

Тогда общее решение для непрерывной модели будет:

$$\boldsymbol{h}(t) = e^{\boldsymbol{A}t} \boldsymbol{h}(0) + \int_0^t{e^{\boldsymbol{A}(t-\tau)}} \boldsymbol{Bx}(\tau) d\tau$$

Шаг дискретизации $`\Delta \Rightarrow \boldsymbol{x_k} = \boldsymbol{x}(k \Delta), \boldsymbol{h_k} = \boldsymbol{h}(k \Delta), 
\boldsymbol{y_k} = \boldsymbol{y}(k\Delta)`$:

$$ \boldsymbol{h_k} = e^{\boldsymbol{A}k\Delta} \boldsymbol{h}(0) + \int_0^{k\Delta}{e^{\boldsymbol{A}(k\Delta-\tau)}} \boldsymbol{Bx}(\tau) d\tau $$

$$ 
\boldsymbol{h_{k+1}} = e^{\boldsymbol{A}\Delta} 
\left[ e^{\boldsymbol{A}k\Delta} \boldsymbol{h}(0) + \int_0^{k\Delta}{e^{\boldsymbol{A}(k\Delta-\tau)}} \boldsymbol{Bx}(\tau) d\tau\right] +
\int_{k\Delta}^{(k+1)\Delta} e^{\boldsymbol{A}\left[(k+1)\Delta-\tau\right]} \boldsymbol{Bx}(\tau) d\tau = $$

Подставляем выражение для $`\boldsymbol{h_k}`$ и учитываем, что $`\boldsymbol{x}=const`$ внутри интервала  $`\Delta`$:

$$ 
= e^{\boldsymbol{A}\Delta} \boldsymbol{h_k} + \left[ \int_0^{\Delta} e^{\boldsymbol{A}\nu} d\nu \right] \boldsymbol{Bx_k} = 
e^{\boldsymbol{A}\Delta} \boldsymbol{h_k} + \frac{1}{\boldsymbol{A}} (e^{\boldsymbol{A}\Delta}-\boldsymbol{I})\boldsymbol{Bx_k}
$$

Таким образом получаем дискретную time-invariant state-space model:

$$\left\{ \begin{array}{lcl}
\boldsymbol{h_k} = \overline{\boldsymbol{A}} \boldsymbol{h_{k-1}} + \overline{\boldsymbol{B}} \boldsymbol{x_k}\\ 
\boldsymbol{y_k} = \boldsymbol{\overline{C} h_k} + \overline{\boldsymbol{D}} \boldsymbol{x_k}\\ 
\\
\boldsymbol{\overline{A}} = e^{\boldsymbol{A}\Delta}\\
\boldsymbol{\overline{B}} = \frac{1}{\boldsymbol{A}} (e^{\boldsymbol{A}\Delta}-\boldsymbol{I})\boldsymbol{B} \approx \Delta \boldsymbol{B}\\
\boldsymbol{\overline{C}} = \boldsymbol{C}\\
\boldsymbol{\overline{D}} = \boldsymbol{D}
\end{array} \right$$

Если в параметре $`\boldsymbol{\overline{B}}`$ разложить экспоненту до первого порядка, происходит очень удачное упрощение, поэтому авторы пренебрегают точностью этого, не самого важного, параметра в пользу уменьшения вычислений:

### Interpretation:

$`\boldsymbol{x_k}`$ - вход модели, 
$`\boldsymbol{y_k}`$ - выход модели, 
$`\boldsymbol{h_k}`$ - скрытое состояние или память модели,
$`\boldsymbol{\overline{A}}`$ - главный из параметров, отвечет за то, как мы преобразуем память с течением времени - параметр запоминания,
$`\boldsymbol{\overline{B}}`$ - параметр преобразования входа,
$`\boldsymbol{\overline{C}}`$ - параметр преобразования выхода,
$`\boldsymbol{\overline{D}}`$ - своего рода **skip connection** или skip параметр,
$`\Delta`$ - шаг дискретизации.

В простейшем случае имеем такие размерности:

$$ \boldsymbol{\overline{A}} (N, N), \\;
\boldsymbol{\overline{B}} (N, 1), \\;
\boldsymbol{\overline{C}} (1, N), \\;
\boldsymbol{\overline{D}} (1, 1), \\;
\boldsymbol{x_k} (1, 1), \\;
\boldsymbol{y_k} (1, 1), \\;
\boldsymbol{h_k}(N, 1), \\;
\Delta = const
$$

## Selective state space model
### Glossary
$`N`$ - размерность скрытого состояния <br/>
$`L`$ - длина входной последовательности <br/>
$`b`$ - размер батча <br/>
$`d`$ - глубина модели <br/>
$`E`$ - коэффициент расширения <br/>
$`d_{in} = Ed`$ - глубина модели в mamba блоке <br/>
$`A,B,C,D`$ - параметры SSM <br/>
$`\Delta`$ - размер шага дискретезации <br/>
$`\Delta_R = \frac{d}{16}`$ - размерность проекции <br/>


### Parametrization
Чтобы модель могла акцентировать внимание на определенных элементах входной последовательности, нужно сделать ее параметры (или часть из них) зависимыми от входа:

$$ \boldsymbol{B} = \boldsymbol{xW_B}, \\; \boldsymbol{C} = \boldsymbol{xW_C}, \\; \Delta = Softplus[\boldsymbol{xW_{\Delta1} W_{\Delta2}}+\Delta_{bias}]$$

$`\boldsymbol{A}`$ и $`\boldsymbol{D}`$ остаются независимыми от входа, но сами становятся параметрами. Параметр $`\boldsymbol{A}`$ будем хранить в логарифмической форме $`\boldsymbol{A_{log}}`$:

$$ \boldsymbol{A} = -\exp^{\boldsymbol{A_{log}}}$$

Таким образом, обучаемые параметры для селективного блока:

$$ 
\boldsymbol{A_{log}}(d_{in}, N), \boldsymbol{W_{B}}(d_{in}, N), \boldsymbol{W_{C}}(d_{in}, N), \boldsymbol{D}(d_{in}), \boldsymbol{W_{\Delta1}}(d_{in}, \Delta_R), \boldsymbol{W_{\Delta2}}(\Delta_R, d_{in}), \Delta_{bias}(d_{in})
$$

Введем сразу остальные параметры, которые будут использоваться в архитектуре:

$$
\boldsymbol{W_{in}}(d, 2d_{in}), \boldsymbol{W_{out}}(d_{in}, d), 
\boldsymbol{W_{emb}}(vocab\\:size, d)=\boldsymbol{W_{vocab}}^T, \boldsymbol{W_{conv1d}}(d_{in}, 1, K)
$$

### Parameters initialization

Каждый из вышеописанных парметров инициаизируется по своему:

$$ 
\boldsymbol{A_{log}} = \ln
\begin{pmatrix}
1 & 2 & 3 & ... & N\\
1 & 2 & 3 & ... & N\\
  &   &...
\end{pmatrix}, \\; \boldsymbol{h_0} = \overline{\boldsymbol{0}}, \\; \boldsymbol{D} = \overline{\boldsymbol{1}}, \\; \Delta_{bias} = Softplus^{-1}\left[Uniform(10^{-3}, 10^{-1}) \right]
$$

Параметр $`W_{conv1d}`$ задается стандартной инициализацией **conv1d** слоя с **bias=True**, тогда как все оставшиеся веса задаются **Linear** слоем с **bias=False**.
