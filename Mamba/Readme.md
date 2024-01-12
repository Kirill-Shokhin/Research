# Mamba or first Selective SSM
![image (7)](https://github.com/Kirill-Shokhin/Research/assets/46619252/b6209055-8de7-43ae-902e-f116735ef9b2)

*Во времена повсеместного заполонения трансформерами, которые пожирали в себя все больше и больше
кремниевых чипов; когда казалось, что лучше уже не будет и за каждый новый токен нужно платить
в квадрате от предыдущих, в эту дремучую зимнюю вьюгу появилась она - **Мамба**.*

## Введение
Трансформеры произвели настоящий фурор во всем Deep Learning и работают они действительно хорошо, но имеют серьезное ограничение по длине входной последовательности из-за квадратичного роста параметров. Подавляющее большинство моделей работают с длиной последовательности менее $`10 000`$,
из-за чего становятся неприменимыми в задачах с большим количеством входных данных. И хотя уже ходили слухи, было бы странно увидеть сильный ИИ, который можно за пару минут заболтать до беспамятства.

Мамба основывается на принципиально другом подходе - state space model (SSM), который хоть и сильно старше трансформера, в контексте DL не показывал достаточную эффективность, особенно в качестве языковой модели. 
Мамба не имеет прямой зависимости по количеству параметров от входа, пересчитывая каждый элемент последовательности циклично, однако, занимет пропорциональное количество памяти. Авторы проверили свое детище на серии моделей только до $`2.8`$ млрд. параметров, что еще мало похоже на **Chatgpt**, но уже утерли нос текущим топам языковых моделей в своей весовой категории.
В этой статье мы детально рассмотрим всю математику новой архитектуры, заметая под ковер преимущества и недостатки.

## General state space model
### Linear continuous time-invariant SSM

Непрерывный вид модели пространства состояний, на которой построена вся идея, выглядит так:

$$ \boldsymbol{\dot h}(t) = \boldsymbol{Ah}(t)+\boldsymbol{Bx}(t) $$

$$\boldsymbol{y}(t) = \boldsymbol{Ch}(t)+\boldsymbol{Dx}(t) $$

С помощью такой модели можно записать дифференциальное уравнение $`N`$-го порядка как $`N`$ уравнений первого порядка в матричном виде, 
где $`\boldsymbol{h}(t)`$ - вектор состояния, содержащий производные по возрастанию порядка от $`0`$ до $`N-1`$, $`x(t)`$- входной сигнал, $`y(t)`$ - выходной сигнал. Таким образом, сложность описываемой системы нелинейно растет от $`N`$.

По-другому на модель можно смотреть так - одномерный сигнал $`x(t)`$ отображается в $`N`$-мерное латентное состояние $`\boldsymbol{h}(t)`$,а затем проецируется в выходной сигнал $`y(t)`$.

Чтобы взаимодествовать с моделью в векторном виде конечной размерности, нужно дискретизовать ее.

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

Таким образом, мы получили простую рекурентную систему, сохранив при этом всю математическую силу пространства состояний. 
Интуицию по стандартной SSM можно получить [здесь](https://srush.github.io/annotated-s4/).

## Selective state space model
Отличительная особенность Мамбы от предыдущих глубоких SSM в этой ветке эволюции состоит в добавлении селективности.
Иначе говоря, мы хотим, чтобы в скрытое состояние $`\boldsymbol{h_k}`$ попадали только значимые из всех $`\boldsymbol{h_i} [i \lt k]`$, а остальные отсеивались. 

### Glossary
$`N`$ - размерность скрытого состояния <br/>
$`L`$ - длина входной последовательности <br/>
$`b`$ - размер батча <br/>
$`d`$ - глубина модели <br/>
$`E=2`$ - коэффициент расширения <br/>
$`d_{in} = Ed`$ - глубина модели в mamba блоке <br/>
$`A,B,C,D`$ - параметры SSM <br/>
$`\Delta`$ - размер шага дискретезации <br/>
$`\Delta_R = \frac{d}{16}`$ - размерность проекции <br/>


### Parametrization
Итак, чтобы модель могла акцентировать внимание на определенных элементах входной последовательности, сделаем три параметра зависимыми от входа:

$$ \boldsymbol{B} = \boldsymbol{xW_B}, \\; \boldsymbol{C} = \boldsymbol{xW_C}, \\; \Delta = Softplus[\boldsymbol{xW_{\Delta1} W_{\Delta2}}+\Delta_{bias}]$$

Параметр $`\Delta`$ управляет балансом между тем, насколько сильно фокусироваться или игнорировать текущий входной сигнал.
Большой $`\Delta`$ сбрасывает состояние $`\boldsymbol{h_k}`$ и фокусируется на текущий вход $`\boldsymbol{x_k}`$, в то время как маленький $`\Delta`$ сохраняет состояние и игнорирует текущий вход.
Параметры $`\boldsymbol{B}`$ и $`\boldsymbol{C}`$ позволяют более тонко контролировать, вводить ли вход $`\boldsymbol{x_k}`$ в состояние $`\boldsymbol{h_k}`$ или состояние в выход $`\boldsymbol{y_k}`$.
$`\boldsymbol{A}`$ и $`\boldsymbol{D}`$ остаются независимыми от входа, но сами становятся параметрами. 

$`\boldsymbol{A}`$ будем хранить в логарифмической форме $`\boldsymbol{A_{log}}`$:

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
\end{pmatrix}, \\; \boldsymbol{D} = \overline{\boldsymbol{1}}, \\; \Delta_{bias} = Softplus^{-1}\left[Uniform(10^{-3}, 10^{-1}) \right]
$$

Параметр $`\boldsymbol{W_{conv1d}}`$ задается стандартной инициализацией **conv1d** слоя с **bias=True**, тогда как все оставшиеся веса задаются **Linear** слоем с **bias=False**.

### Selective SSM inference with Hardware-aware State Expansion

*Рисунок 1: Устройство Selective SSM блока ([Mamba](https://arxiv.org/abs/2312.00752)).*
![SSSM](https://github.com/Kirill-Shokhin/Research/assets/46619252/8485b8db-b090-4324-98df-ed6f92badfe9)


$$\boldsymbol{x}(b, L, d_{in}), \boldsymbol{h_t}(b, d_{in}, N) \rightarrow \boldsymbol{y}(b, L, d_{in})$$

Для ускорения вычислений авторы разделили инференс selective SSM блока на два этапа - сначала подготовка (трехмерных массивов) на обычной (медленной) памяти видеокарты, затем дискретизация и вычисление рекурсии (четырехмерных массивов) в быстрой памяти видеокарты:

#### 1) Подготовка (GPU HBM):
Возвращение $`\boldsymbol{A}`$ в человеческий вид:

$$\boldsymbol{A}(d_{in}, N) = -\exp^{\boldsymbol{A_{log}}}$$

Проекция входа:
     
 $$\begin{array}{ccc}
 \boldsymbol{B}(b, L, N) = \boldsymbol{xW_B}\\
 \boldsymbol{C}(b, L, N) = \boldsymbol{xW_C}\\
 \Delta(b, L, d_{in}) = Softplus[\boldsymbol{xW_{\Delta1} W_{\Delta2}}+\Delta_{bias}]
 \end{array}$$

#### 2) Selective scan (GPU SRAM):
   
Инициализация скрытого состояния: 

$$\boldsymbol{h_{-1}} = \overline{\boldsymbol{0}}$$

Дискретизация:
     
$$\begin{array}{ccc}
\boldsymbol{\overline{A}}(b, L, d_{in}, N) = e^{\Delta \boldsymbol{A}}\\
\boldsymbol{\overline{B}x}(b, L, d_{in}, N) = \Delta \boldsymbol{Bx}
\end{array}$$

В цикле по $`t`$ вдоль оси $`L`$ (по каждому токену) пересчет всех скрытых состояний $`\boldsymbol{h}`$ и соответсвующих им выходов $`\boldsymbol{y}`$:
   
$$\begin{array}{lcl}
\boldsymbol{h_t} = \overline{\boldsymbol{A_t}} \boldsymbol{h_{t-1}} + \boldsymbol{(\overline{B} x)_t}\\ 
\boldsymbol{y_t} = \boldsymbol{C_t h_t} + \boldsymbol{Dx_t}\\ 
\end{array}$$

## Архитектура Mamba

![Mamba схемы](https://github.com/Kirill-Shokhin/Research/assets/46619252/15b732bd-da9a-44db-85a6-ed3e77faf6e3)
*Рисунок 2: Устройство архитектуры Mamba*

$`\boldsymbol{x}(b,L,d), \boldsymbol{x_{in}}(b,L,d_{in}), \boldsymbol{W_{in}}(d,d_{in})`$ - для каждой ветки, $`\sigma = SiLU`$

### Mamba
Устройство архитектуры Мамбы не сильно отличается от трансформерной: 
1) На входе имеем последовательность длиной $`L`$, которая может представлять из себя хоть текстовые токены, хоть элементы изображения.
2) Векторизуем элементы последовательности матрицой эмбеддингов $`\boldsymbol{W_{emb}}`$, получая тот самый $`\boldsymbol{x}(b, L, d)`$.
3) Прогоняем его через $`n_{layers}`$ мамба-слоев, сохраняя при этом размерность. 
4) Возвращаем размерность $`(b, L, vocab\;size)`$ матричным умножением на $`\boldsymbol{W_{vocab}=W_{emb}^T}`$ - той же матрицей, что и на входе.
5) И, наконец, получаем вероятности для каждого токена по словарю.

### Mamba Layer
Слой Мамба представляет из себя:
1) Нормализацию по слою
2) Непосредственно сам мамба блок 
3) Skip connection

### Mamba Block
Принцип блока основан на **gated MLP**, который при помощи дополнительной ветки с линейным слоем, 
активацией и последующим **Element-wise** умножением может управлять потоком информации основной ветки, 
определяя какая часть должна быть сохранена, а какая подавлена.

По основной же ветке идет, так называемый, **inverted bottleneck**:

* Расширение $`(\boldsymbol{W_{in}}) \rightarrow`$ **depthwise convolution** (в данном случае одномерная) $`\rightarrow`$ проекция $`(\boldsymbol{W_{out}})`$,

с добавлением активации и основного блока - **selective SSM** из предыдущего раздела.

## Заключение
