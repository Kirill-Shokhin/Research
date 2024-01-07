# Deep State Space models

### Linear continuous time-invariant state-space model:

$$ \boldsymbol{\dot x}(t) = \boldsymbol{Ax}(t)+\boldsymbol{Bu}(t) $$

$$\boldsymbol{y}(t) = \boldsymbol{Cx}(t)+\boldsymbol{Du}(t) $$

Применительно к DL параметр полагают $`\boldsymbol{D}=0`$, поэтому далее везде опускаем это слагаемое.

### Discretization of linear state space model:
Умножим первое уравнение на $`e^{-\boldsymbol{A}t}`$:

$$ e^{-\boldsymbol{A}t} \boldsymbol{\dot x}(t) - e^{-\boldsymbol{A}t} \boldsymbol{Ax}(t) = e^{-\boldsymbol{A}t} \boldsymbol{Bu}(t)$$

$$ \frac{d}{dt} (e^{-\boldsymbol{A}t} \boldsymbol{x}(t)) = e^{-\boldsymbol{A}t} \boldsymbol{Bu}(t)$$

Тогда общее решение для непрерывной модели будет:

$$\boldsymbol{x}(t) = e^{\boldsymbol{A}t} \boldsymbol{x}(0) + \int_0^t{e^{\boldsymbol{A}(t-\tau)}} \boldsymbol{Bu}(\tau) d\tau$$

Шаг дискретизации $`\Delta \Rightarrow \boldsymbol{u_k} = \boldsymbol{u}(k \Delta), \boldsymbol{x_k} = \boldsymbol{x}(k \Delta), 
\boldsymbol{y_k} = \boldsymbol{y}(k\Delta)`$:

$$ \boldsymbol{x_k} = e^{\boldsymbol{A}k\Delta} \boldsymbol{x}(0) + \int_0^{k\Delta}{e^{\boldsymbol{A}(k\Delta-\tau)}} \boldsymbol{Bu}(\tau) d\tau $$

$$ 
\boldsymbol{x_{k+1}} = e^{\boldsymbol{A}\Delta} 
\left[ e^{\boldsymbol{A}k\Delta} \boldsymbol{x}(0) + \int_0^{k\Delta}{e^{\boldsymbol{A}(k\Delta-\tau)}} \boldsymbol{Bu}(\tau) d\tau\right] +
\int_{k\Delta}^{(k+1)\Delta} e^{\boldsymbol{A}\left[(k+1)\Delta-\tau\right]} \boldsymbol{Bu}(\tau) d\tau = $$

Подставляем выражение для $`\boldsymbol{x_k}`$ и учитываем, что $`\boldsymbol{u}=const`$ внутри интервала  $`\Delta`$:

$$ 
= e^{\boldsymbol{A}\Delta} \boldsymbol{x_k} + \left[ \int_0^{\Delta} e^{\boldsymbol{A}\nu} d\nu \right] \boldsymbol{Bu_k} = 
e^{\boldsymbol{A}\Delta} \boldsymbol{x_k} + \frac{1}{\boldsymbol{A}} (e^{\boldsymbol{A}\Delta}-\boldsymbol{I})\boldsymbol{Bu_k}
$$

Таким образом получаем дискретную time-invariant state-space model:

$$\left\{ \begin{array}{lcl}
\boldsymbol{x_k} = \overline{\boldsymbol{A}} \boldsymbol{x_{k-1}} + \overline{\boldsymbol{B}} \boldsymbol{u_k}\\ 
\boldsymbol{y_k} = \boldsymbol{\overline{C} x_k}\\ 
\\
\boldsymbol{\overline{A}} = e^{\boldsymbol{A}\Delta}\\
\boldsymbol{\overline{B}} = \frac{1}{\boldsymbol{A}} (e^{\boldsymbol{A}\Delta}-\boldsymbol{I})\boldsymbol{B}\\
\boldsymbol{\overline{C}} = \boldsymbol{C}\\
\end{array} \right$$

### Bilinear transform:
Для упрощения вычислений матричной экспоненты применяем следующую апроксимацию:

$$ e^{\boldsymbol{A}\Delta} \approx \left(\boldsymbol{I} - \frac{1}{2} \boldsymbol{A}\Delta \right)^{-1} 
\left(\boldsymbol{I} + \frac{1}{2} \boldsymbol{A}\Delta \right)$$

Таким образом, параметры принимают следующий вид:

$$ 
\boldsymbol{\overline{A}} = \dfrac{\left(\boldsymbol{I} + \frac{1}{2} \boldsymbol{A}\Delta \right)}
{\left(\boldsymbol{I} - \frac{1}{2} \boldsymbol{A}\Delta \right)}, \\;\\;\\;
\boldsymbol{\overline{B}} = \dfrac{\Delta \boldsymbol{B}}{\left(\boldsymbol{I} - \frac{1}{2} \boldsymbol{A}\Delta \right)}, \\;\\;\\;
\boldsymbol{\overline{C}} = \boldsymbol{C}
$$