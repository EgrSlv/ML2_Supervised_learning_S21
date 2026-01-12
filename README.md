# Linear regression model

Данный проект посвящен обучению с учителем, в частности линейным моделям, методам регуляризации, переобучению, недообучению и метрикам оценки качества.

## Contents

- [Linear regression model](#linear-regression-model)
  - [Contents](#contents)
  - [Chapter I. Preamble](#chapter-i-preamble)
  - [Chapter II. Introduction](#chapter-ii-introduction)
    - [Regression problem](#regression-problem)
    - [Linear Regression](#linear-regression)
    - [Gradient Descent](#gradient-descent)
    - [Overfitting and underfitting](#overfitting-and-underfitting)
    - [Regularization](#regularization)
    - [Quality metrics](#quality-metrics)
    - [Alternative linear regression problem formulation](#alternative-linear-regression-problem-formulation)
  - [Chapter III. Goal](#chapter-iii-goal)
  - [Chapter IV. Instructions](#chapter-iv-instructions)
  - [Chapter V. Task](#chapter-v-task)
    - [Submission](#submission)


## Chapter I. Preamble

В прошлом проекте мы обсуждали, что такое машинное обучение и какие проблемы решает эта область науки. А также подробно рассмотрели примеры наиболее сложных задач и на какие группы их можно разделить. Цель открытого проекта — одна из таких задач, и мы познакомимся с первой из них — линейной моделью.

Но прежде чем мы начнём, я хотел бы кратко описать, как обычно подходят к описанию всех инструментов. Затем мы рассмотрим формулировку проблемы с математической точки зрения. А основным инструментом для решения этих задач будет следующий алгоритм:

Мы ограничиваем набор возможных решений определенным множеством (множеством функций).

Например, мы хотим предсказать температуру y на основе давления x, используя всего одно наблюдение. Мы ограничиваемся набором линейных функций:

$$ 
y \approx \hat{y} = f(x) = wx
$$

Как правило, каждое решение в этом множестве может быть однозначно определено параметрами. В нашем примере это параметр *a*.

Мы выбираем функцию потерь, которая наглядно демонстрирует наше стремление найти решение, удовлетворяющее цели задачи.

В нашем примере это стандартное отклонение, которое принимает высокие значения, когда наше предсказание далеко от истинного значения.

$$ 
L \left(y, \hat{y} \right) = \left( y - \hat{y} \right)^2
$$

Таким образом, поиск оптимального решения сводится к нахождению решения, для которого функция потерь минимальна. Однако, поскольку функция будет зависеть от параметра, можно сказать, что цель состоит в том, чтобы найти такие параметры, для которых функция потерь минимальна.

В нашем примере формальная задача будет выглядеть следующим образом:

$$ 
\arg \min_{\hat{y}} L \left( y, \hat{y} \right) = \arg \min_{w} L(y, wx)
$$

Тот факт, что функция потерь зависит от параметров, позволяет нам использовать производные и методы оптимизации (включая градиентный спуск) для нахождения оптимальных значений параметров.

В нашем примере:

$$ 
\frac{\partial L}{\partial w}=-2w(y-wx)=0 \Rightarrow w=\lbrace \frac{y}{x}, 0 \rbrace
$$

Часто бывает возможно взглянуть на одну и ту же математическую задачу с другой стороны, решив её в альтернативной формулировке. Это откроет новые смыслы в решении, которые помогут вам лучше понять детали. Поэтому мы настоятельно рекомендуем вам не зацикливаться на материалах, упомянутых в этом курсе, и периодически совершенствовать своё понимание того или иного подхода.

В этой главе мы обсудим проблемы регрессии, которые ограничивают набор возможных решений в линейных моделях. Наряду с самой моделью, мы рассмотрим вопросы, тесно связанные с процессом ее разработки: определения переобучения/недообучения, способы борьбы с ними и методы оценки качества моделей.

## Chapter II. Introduction

### Regression problem

Предположим, у нас есть набор входных данных *X*, которым соответствуют выходные данные *y*. Цель состоит в том, чтобы найти функцию отображения f из *X* в *y*. В задачах регрессии $ y ∈ R $ и наш $ X ∈ R^n $. Другими словами:

$$
f: X \rightarrow y, \text{где } y \in \mathbb{R} \text{ это цель, } X \text{ тренировочные данные }
$$

Существует множество примеров реальных задач регрессионного анализа. Рассмотрим наиболее популярные из них:  
 * Прогнозирование возраста зрителя, смотрящего теле- или видеоканалы.  
 * Прогнозирование амплитуды, частоты и других характеристик сердцебиения.  
 * Прогнозирование количества такси, необходимых в зависимости от погодных условий.  
 * Прогнозирование температуры в любой точке внутри здания с использованием данных о погоде, времени, датчиков дверей и т. д.  

В случае линейной регрессии ключевым свойством является предположение, что ожидаемое значение выходного сигнала является линейной функцией входного сигнала: $f(\mathbf{x}) = \mathbf{w}^{\boldsymbol{\top}} \mathbf{x} + b$. Это упрощает подгонку модели к данным и ее интерпретацию.

Для регрессии наиболее распространенным выбором является использование квадратичной функции потерь, или функции потерь типа «l2», или MSE — среднеквадратичной ошибки:

$$
l_2 \left(y, \hat{y} \right) = \left( y - \hat{y} \right)^2
$$

где $ \hat y $ — это предсказание модели, а $ y $ — истинное значение. Мы используем функцию потерь для минимизации ошибки и подгонки модели. Давайте рассмотрим модель линейной регрессии.

### Linear Regression

Одним из наиболее широко используемых, но не всегда эффективных алгоритмов для решения задач регрессии является модель линейной регрессии. Главная цель этой модели — вычислить линейную функцию обучающего набора данных с наименьшей ошибкой.

Мы можем подогнать под данные модель линейной регресси по следующей формуле:

$$
f(x;\boldsymbol{\theta}) = b + \mathbf{w} x
$$

где:
* $ \mathbf{w} $ — это наклон (вес, коэффициент)
* $ b $ — это смещение (свободный член, отступ по оси ординат)
* $ \boldsymbol{\theta} = (\mathbf{w}, b) $ — все параметры модели ($\theta$ — тетта).

Регулируя $\boldsymbol{\theta}$, мы можем минимизировать сумму квадратов ошибок до тех пор, пока не найдем решение методом наименьших квадратов:

$$
\hat{\boldsymbol{\theta}} = \arg \min_{\boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta})
$$

$$
\text{MSE}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{n=1}^{N} \left( y_n - f(x_n; \boldsymbol{\theta})\right)^2
$$

где MSE — среднеквадратичная ошибка.

Если говорить формально, модель линейной регрессии можно записать следующим образом:

$$ 
f(x;\boldsymbol{\theta}) = \mathbf{w}^{\boldsymbol{\top}} \mathbf{x} + b + \epsilon = \sum_{i=1}^D w_i x_i + b + \epsilon
$$

где:
* $ \mathbf{w}^{\boldsymbol{\top}} \mathbf{x}$ является скалярным произведением между входным вектором $\mathbf{x}$ и вектором весов модели $\mathbf{w}$
* $ b $ — это смещение (свободный член, отступ по оси ординат)
* $ \epsilon $ — это остаточная ошибка между нашими линейными прогнозами и истинным значением.

Математически нашу задачу можно определить как минимизацию среднеквадратичной ошибки (MSE). Для решения таких задач можно использовать методы оптимизации, такие как градиентный спуск или метод Ньютона, или даже записать аналитическое решение, включающее обращение матрицы (после выполнения первой задачи, которую мы обсудим позже в этой главе — вы обнаружите, что обращение матрицы — это ресурсоемкая вычислительная операция). Ваша первая задача будет заключаться в нахождении аналитического решения задачи регрессии.

### Gradient Descent

Общеизвестно, что градиент — это направление самого быстрого увеличения значения функции, а в противоположном случае антиградиент — это направление самого быстрого уменьшения. Таким образом, градиентный спуск — это поиск локального минимума дифференцируемой функции. А поскольку наша главная цель почти в любой задаче машинного обучения — уменьшить значение функции потерь, градиентный спуск идеально подходит для этой цели.

Для этого нам следует итеративно повторять следующие шаги:
1. Вычислите значение градиента:  
    $$ 
    \frac{\partial L}{\partial \boldsymbol{\theta}} = \nabla L = \left( \begin{array}{c} \frac{\partial L}{\partial \theta_1} \\ \dots \\ \frac{\partial L}{\partial \theta_n} \end{array} \right)
    $$

2. Обновите значения параметров, используя формулу:  
    $$ 
    \boldsymbol{\theta}^{i+1} = \boldsymbol{\theta}^i - \gamma \nabla L \left( \boldsymbol{\theta}^i \right)
    $$
    где 
    * $\gamma$ — это скорость обучения, определяющая, насколько далеко мы должны продвинуться, и где 0 обычно определяется случайным образом.

![](misc/images/gradient_descent.jpg)

Существует множество модификаций градиентного спуска для получения более точных результатов и ускорения вычислений.
Рассмотрим основную модификацию — стохастический градиентный спуск. Преимущество этого метода заключается в том, что мы стохастически выбираем объект (или набор объектов) и вычисляем градиент только для этого объекта. Это помогает значительно сэкономить время, но качественная эффективность существенно не снижается.

Мы рекомендуем вам ознакомиться с другими модификациями GD, такими как методы Adagrad, RMSProp, Adam и Momentum, а также с методами вторичного порядка, такими как метод Ньютона.

### Overfitting and underfitting

Основная проблема метода максимального правдоподобия (MLE) заключается в том, что он пытается выбрать параметры, которые минимизируют потери на обучающем наборе данных, но это может не привести к созданию модели с низкими потерями на будущих данных. Это называется переобучением.

Поэтому при построении моделей с высокой гибкостью необходимо избегать переобучения, то есть следует избегать попыток моделировать каждое небольшое изменение входных данных, поскольку это, скорее всего, шум, а не истинный сигнал.

Это показано на рисунке ниже, где видно, что использование полинома высокой степени приводит к кривой, которая очень «извилистая». Истинная функция вряд ли будет иметь такие экстремальные колебания. Следовательно, использование такой модели может привести к точным прогнозам будущих результатов.

![](misc/images/graphics.jpg)

Недообученная модель — это противоположность переобученной модели — модель, которая не изучает зависимости из обучающего набора данных или изучает их неправильным образом. Это снова приводит к низкой производительности на тестовом наборе данных. Например, нам нужно классифицировать рукописные цифры, и оказывается, что все изображения цифр 8 имеют 67 темных пикселей, в то время как все остальные примеры их не имеют.

Модель может решить использовать эту случайную закономерность и, таким образом, правильно классифицировать все обучающие примеры восьмерок, не изучив истинные закономерности. А когда мы передадим новые изображения восьмерок, написанные в другом масштабе, мы получим очень плохие результаты, потому что количество темных пикселей будет другим. Модель не изучит необходимые закономерности.

Другими словами, нам нужно скорректировать модель, чтобы найти баланс между этими двумя случаями. Это также известно как компромисс между смещением и дисперсией. Ошибка, возникающая при подгонке и оценке модели, может быть разделена на две части (на самом деле на три, но на третью мы повлиять не можем): смещение и дисперсия.  
Чтобы понять, что означает каждый тип ошибки, см. рисунок из [Википедии](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff):

![](misc/images/low_low.jpg)
![](misc/images/high_low.jpg)

![](misc/images/low_high.jpg)
![](misc/images/high_high.jpg)

> By Bernhard Thiery — Own work, CC BY-SA 3.0

* «Ошибка смещения — это ошибка, возникающая из-за неверных предположений в алгоритме обучения. Высокое смещение может привести к тому, что алгоритм упустит важные связи между признаками и целевыми выходными данными (недообучение)».  
* «Дисперсия — это ошибка, возникающая из-за чувствительности к небольшим колебаниям в обучающем наборе данных. Высокая дисперсия может привести к тому, что алгоритм будет моделировать случайный шум в обучающих данных (переобучение)».  

К счастью, большинство моделей в машинном обучении сталкиваются с проблемой смещения и могут иметь проблемы, в основном, с дисперсией. Но существует множество специальных методов для уменьшения переобучения модели. Здесь мы хотим представить один из них, который называется регуляризацией.

### Regularization

Обычно регуляризация происходит за счет того, что она просто расходует аддитивную составляющую в функции потерь, которая зависит от параметров модели.

$$ 
\min_{\boldsymbol{\theta}} \sum_{i=1}^N L \left( f(x_i, \boldsymbol{\theta}), y_i \right) + \lambda R(\boldsymbol{\theta})
$$

Идея этого дополнения заключается в том, что модели не пытаются изучать сложные закономерности, а стремятся найти решение. Другими словами, если модель начинает переобучаться в процессе поиска решения, то добавление в функцию ошибки начнет увеличиваться, заставляя модель прекратить обучение.

Для линейной регрессии обычно используются 2 варианта этих дополнений:
* L1 — тогда линейная модель называется моделью Lasso.  
    $$ 
    R(\boldsymbol{\theta}) = \| \boldsymbol{\theta}\|_1 = \sum_{i=1}^d |\theta_i|
    $$

* L2 — тогда линейная модель называется моделью Риджа.
    $$ 
    R(\boldsymbol{\theta}) = \| \boldsymbol{\theta}\|_2^2 = \sum_{i=1}^d \theta_i^2
    $$

* В Sklearn также есть отдельный класс для случаев, когда эти две регуляризации объединены, и такая модель называется Elastic.

Подумайте, как изменится алгоритм поиска решения для линейной модели, если добавить L1 и L2 регуляризацию к функции потерь. Представьте свой ответ в рамках проекта.

### Quality metrics

Давайте рассмотрим метрики качества для оценки задач регрессии. Мы уже изучили среднеквадратичную ошибку. А после изучения определения регуляризации **L1** вы уже знаете, что такое средняя абсолютная ошибка. Но эта метрика не является относительной. Что это значит?

Допустим, мы решаем задачу расчета дохода клиентов, поэтому наша целевая переменная является непрерывной. А средняя абсолютная ошибка (MAE) нашей модели равна 5000. Как мы можем оценить, достаточно ли хороша наша модель?

Сделайте перерыв и подумайте несколько минут. Здесь мы обсудим несколько способов решения этой проблемы. Первый и самый простой — сравнить нашу модель с прогнозами, полученными с помощью нативных моделей. Например, мы могли бы найти среднее значение нашей целевой переменной и установить его в качестве прогноза для тестовой выборки для целого набора клиентов. Наконец, сравните MAE нашей модели и прогноза, полученного с помощью нативных моделей. Здесь мы рекомендуем найти наилучший прогноз, полученный с помощью нативных моделей, по показателям MSE и MAE. Также подумайте о том, как мы могли бы улучшить прогноз, полученный с помощью нативных моделей, для повышения его качества.

Второй способ, который мы здесь обсудим, — это преобразование относительных показателей в абсолютные. Средняя абсолютная процентная ошибка (MAPE) — один из широко используемых показателей в задачах регрессионного анализа. MAPE — это сумма индивидуальных абсолютных ошибок, деленная на спрос (за каждый период отдельно). Это среднее значение процентных ошибок. По сравнению с MAE, MAPE имеет точный диапазон. Для лучшего понимания рассмотрим несколько примеров и измерим различные показатели, а также рекомендуем определить MSLE и коэффициент $R^2$.

### Alternative linear regression problem formulation

Рассмотрим линейную регрессию с точки зрения теории вероятностей, и мы сможем переписать модель, связав линейную регрессию с гауссовыми распределениями, в следующей форме:

$$ 
p(y|\text{ }\mathbf{x},\boldsymbol{\theta}) = \mathcal{N}(y|\text{ }μ(\mathbf{x}),σ^2 (\mathbf{x}))
$$

где  
* $ N(µ, \sigma^2) $ — это **гауссово** или **нормальное** распределение
* µ — **среднее значение**
* $ \sigma^2 $ — **дисперсия**.

Это ясно показывает, что модель представляет собой **условную функцию плотности вероятности**. В простейшем случае мы предполагаем, что µ является линейной функцией от *x*, так что $ µ= w^{\boldsymbol{\top}}x $, и что шум фиксирован, $ \sigma^2(x) = \sigma^2 $.

В данном случае $ \theta = (w,\sigma^2) $ являются **параметрами** модели.

Например, предположим, что входные данные одномерные. Ожидаемый ответ можно записать следующим образом:

$$ 
µ( \boldsymbol{x})= \boldsymbol{w_0} + \boldsymbol{w_1x} = \boldsymbol{w^{\boldsymbol{\top}}x}
$$

где $ \boldsymbol{w_0} $ — свободный член или член смещения, $ \boldsymbol{w_1} $ — наклон, и мы определили вектор $\boldsymbol{x = (1, x)}$.

Давайте попробуем разобраться, как определить параметры модели линейной регрессии.

Распространенный способ оценки параметров статистической модели — это вычисление оценки максимального правдоподобия (MLE).  
**MLE** — это метод оценки неизвестного параметра путем максимизации функции правдоподобия, которую можно определить следующим образом:

$$ 
\hat \theta = arg \max\limits_{\theta} L(\theta) = arg \max\limits_{\theta} \prod_{i=1}^l P(y_i|x_i, \theta)
$$

где $ L(\theta) $ — функция правдоподобия (Likelihood function).

Обычно предполагается, что обучающие примеры независимы и имеют одинаковое распределение. Это означает, что логарифмическую функцию правдоподобия можно записать следующим образом:

$$ 
\ell\left(\boldsymbol{\theta}\right)\triangleq\log{p}\left(\mathcal{D}\left|{\boldsymbol{\theta}}\right.\right)=\sum_{i=1}^{N}\log{p\left(y_i\left|\boldsymbol{x_i},\ {\boldsymbol{\theta}}\right.\right)}
$$

Вместо максимизации логарифма правдоподобия мы можем, эквивалентно, минимизировать отрицательный логарифм правдоподобия, или NLL:

$$ 
\text{NLL}\left(\boldsymbol{\theta}\right)\triangleq-\sum_{i=1}^{N}\log{p\left(y_i\left|\boldsymbol{x_i},\boldsymbol{\theta}\right.\right)}
$$

Формулировка **NLL** иногда более удобна, поскольку многие пакеты программного обеспечения для оптимизации предназначены для поиска минимумов функций, а не максимумов. Теперь применим метод MLE к линейной регрессии. Подставив определение гауссова распределения в приведенное выше выражение, мы получим, что логарифмическая функция правдоподобия задается следующим образом:

$$ 
\ell(\boldsymbol{\theta}) = \sum_{i=1}^N \log\left[ \left(\frac{1}{2\pi\sigma^2}\right)^{\frac{1}{2}} \exp\left( -\frac{1}{2\sigma^2} (y_i - \mathbf{w}^{\boldsymbol{\top}}\mathbf{x}_i)^2 \right) \right] = \frac{-1}{2\sigma^2} \text{RSS}(\mathbf{w}) - \frac{N}{2} \log(2\pi\sigma^2)
$$  

RSS (residual sum of squares) расшифровывается как сумма квадратов остатков и определяется следующим образом:

$$ 
\text{RSS}(\mathbf{w}) \triangleq \sum_{i=1}^{N}(y_i-\mathbf{w}^{\boldsymbol{\top}}\mathbf{x}_i)^2
$$

RSS также называется суммой квадратов ошибок (sum of squared errors - SSE), а отношение SSE/N — среднеквадратичной ошибкой (**MSE**). Её также можно представить как квадрат нормы $ \boldsymbol{\ell^2} $ вектора остаточных ошибок:

$$ 
\text{RSS}(\mathbf{w}) = \|\boldsymbol{\epsilon}\|_2^2 = \sum_{i=1}^N\epsilon_i^2
$$
где $ \boldsymbol{\epsilon_i = (y_i-\mathbf{w}^{\boldsymbol{\top}}\mathbf{x}_i)} $.

Линейная регрессия может использоваться для моделирования нелинейных зависимостей путем замены $ \boldsymbol{x} $ нелинейной функцией входных данных $ \boldsymbol{\phi(x)} $. То есть, мы используем

$$ 
p(y|\text{ }\mathbf{x}, \boldsymbol{\theta}) = \boldsymbol{\mathcal{N}}\left(y|\text{ }\mathbf{w}^{\boldsymbol{\top}}\boldsymbol{\phi}\mathbf{(x)}, \sigma^2\right)
$$

где могут быть степени признаков:  

$ \boldsymbol{\phi(x) = [1, x, x^2, ..., x^d]} $

Оценка методом максимального правдоподобия может привести к переобучению. В некоторых случаях наша задача — минимизация RSS — может иметь бесконечное количество решений. Можете ли вы привести несколько примеров таких случаев? Сделайте перерыв и подумайте. Ответ: ну, несколько распространенных случаев для этой проблемы — признаки с высокой корреляцией и случаи, когда размерность признаков превышает размерность объекта. Затем возникают проблемы с огромными значениями весов и переобучением. Для решения этой проблемы мы можем добавить параметры регуляризации в наше уравнение минимизации. Существуют два основных параметра регуляризации: $ \mathbf{\ell1} $ и $ \mathbf{\ell2} $.

$$ l1 = ||w||_1 = \sum_{l=1}^d |w_i| $$

$$ l2 = ||w||_2 = \sum_{l=1}^d w_i^2 $$

С точки зрения теории вероятностей, регуляризацию можно определить с помощью оценки MAP (Maximum A Posterior estimate) с гауссовым априорным распределением весов с нулевым средним.
$$ 
p(\boldsymbol{w}) = \mathcal{N}(\boldsymbol{w}|\text{ }\boldsymbol{0}, \lambda^{-1}\boldsymbol{\mathbf{I}})
$$

Это называется гребенчатой ​​регрессией (**Ridge-регрессия**). Более конкретно, мы вычисляем оценку MAP следующим образом:
$$
\boldsymbol{\hat{w}}_{\rm{map}} = \arg\min\frac{1}{2\sigma^2}(\boldsymbol{y - \mathbf{X}w})^{\boldsymbol{\top}}(\boldsymbol{y - \mathbf{X}w}) + \frac{1}{2\tau^2}\boldsymbol{w}^{\boldsymbol{\top}}\boldsymbol{w}=\argmin{\text{RSS}}(\boldsymbol{w}) + \lambda \|\boldsymbol{w}\|_2^2
$$  

где $ \lambda\triangleq \LARGE\frac{\sigma^2}{\tau^2} $ пропорционально силе априорного распределения, и $ \displaystyle \|\boldsymbol{w}\|_2 \triangleq \sqrt{\sum_{d=1}^D|w_d|^2} = \sqrt{\boldsymbol{w}^{\boldsymbol{\top}}\boldsymbol{w}} $

является $ \mathcal{\ell 2} $ нормой вектора $ \boldsymbol{w} $. Таким образом, мы штрафуем веса, которые становятся слишком большими. В общем случае этот метод называется $ \mathcal{\ell 2} $ регуляризацией или уменьшением веса и широко используется.

> Source: [Kevin Murphy, Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html)

## Chapter III. Goal

Цель данного задания — получить глубокое понимание модели линейной регрессии.

## Chapter IV. Instructions

* Мы используем Python 3, поскольку это единственная допустимая версия Python.  
* Для обучения алгоритмов глубокого обучения вы можете попробовать [Google Colab](https://colab.research.google.com). Он предоставляет ядра (среду выполнения) с GPU бесплатно, что быстрее, чем CPU для таких задач.  
* Стандарт не применяется к этому проекту. Однако вас просят быть ясными и структурированными в проектировании исходного кода.  
* Храните наборы данных в подпапке data.  

## Chapter V. Task

We will continue our practice with a problem from Kaggle.com. In this chapter, we will implement all the models described above. Measure quality metrics on training and test parts. Detect and regularize overfitted models. And dive deeper with native model estimation and comparison.

Решение приведено в файле [`ML2_SupervisedLearning.ipynb`](ML2_SupervisedLearning.ipynb)  
1. Answer the questions
   1. Derive an analytical solution to the regression problem. Use a vector form of the equation.
   2. What changes in the solution when L1 and L2 regularizations are added to the loss function.
   3. Explain why L1 regularization is often used to select features. Why are there many weights equal to 0 after the model is fit?
   4. Explain how you can use the same models (Linear regression, Ridge, etc.) but make it possible to fit nonlinear dependencies.

2. Introduction — make all the preprocessing staff from the previous lesson
   1. Import libraries. 
   2. Read Train and Test Parts.

3. Intro data analysis part 2
   1. Let's generate additional features for better model quality. Consider a column called "Features". It consists of a list of highlights of the current flat. 
   2. Remove unused symbols ([,], ', ", and space) from the column.
   3. Get all values in each list and collect the result in one huge list for the whole dataset. You can use DataFrame.iterrows().
   4. How many unique values does a result list contain?
   5. Let's get acquainted with the new library — Collections. With this package you could effectively get quantity statistics about your data. 
   6. Count the most popular functions from our huge list and take the top 20 for this moment.
   7. If everything is correct, you should get next values:  'Elevator', 'CatsAllowed', 'HardwoodFloors', 'DogsAllowed', 'Doorman', 'Dishwasher', 'NoFee', 'LaundryinBuilding', 'FitnessCenter', 'Pre-War', 'LaundryinUnit', 'RoofDeck', 'OutdoorSpace', 'DiningRoom', 'HighSpeedInternet', 'Balcony', 'SwimmingPool', 'LaundryInBuilding', 'NewConstruction', 'Terrace'.
   8. Now create 20 new features based on the top 20 values: 1 if the value is in the "Feature" column, otherwise 0.
   9. Extend our feature set with 'bathrooms', 'bedrooms' and create a special variable feature_list with all feature names. Now we have 22 values. All models should be trained on these 22 features.

4. Models implementation — Linear regression
   1. Implement a Python class for a linear regression algorithm with two basic methods — fit and predict. Use stochastic gradient descent (SGD) to find optimal model weights. For better understanding, we recommend implementing separate versions of the algorithm with the analytical solution and non-stochastic gradient descent under the hood.
   2. What is determenistic model? Make SGD determenistic.
   3. Define the R squared (R2) coefficient and implement a function to calculate it.
   4. Make predictions with your algorithm and estimate the model with MAE, RMSE and R2 metrics.
   5. Initialize LinearRegression() from sklearn.linear_model, fit the model, and predict the training and test parts as in the previous lesson.
   6. Compare the quality metrics and make sure the difference is small (between your implementations and sklearn).
   7. Store the metrics as in the previous lesson in a table with columns model, train, test for MAE table, RMSE table, and R2 coefficient.

5. Regularized models implementation — Ridge, Lasso, ElasticNet    
   1. Implement Ridge, Lasso, ElasticNet algorithms: extend the loss function with L2, L1 and both regularizations accordingly.
   2. Make predictions with your algorithm and estimate the model with MAE, RMSE and R2 metrics.
   3. Initialize Ridge(), Lasso(), and ElasticNet() from sklearn.linear_model, fit the model, and make predictions for the training and test samples as in the previous lesson.
   4. Compare quality metrics and make sure the difference is small (between your implementations and sklearn).
   5. Store the metrics as in the previous lesson in a table with columns model, train, test for MAE table, RMSE table, and R2 coefficient.
   
6. Feature normalization
   1. First, write several examples of why and where feature normalization is mandatory and vice versa.
   2. Let's consider the first of the classical normalization methods — MinMaxScaler. Write a mathematical formula for this method.
   3. Implement your own function or class for MinMaxScaler feature normalization.
   4. Initialize MinMaxScaler() from sklearn.preprocessing.
   5. Compare the feature normalization with your own method and with sklearn.
   6. Repeat the steps from b to e for another normalization method StandardScaler.

7. Fit custom and sklearn models with normalized data
   1. Fit all models — Linear Regression, Ridge, Lasso, and ElasticNet — with MinMaxScaler.
   2. Fit all models — Linear Regression, Ridge, Lasso, and ElasticNet — with StandardScaler.
   3. Add all results to our dataframe with metrics on samples.

8. Overfit models
   1. Let's look at an overfitted model in practice. From theory, you know that polynomial regression is easy to overfit. So let's create a toy example and see how regularization works in real life.
   2. In the previous lesson, we created polynomial features with degree 10. Here we repeat these steps from the previous lesson, remembering that we have only 2 basic features — 'bathrooms' and 'bedrooms'.
   3. And train and fit all our implemented algorithms — Linear Regression, Ridge, Lasso, and ElasticNet — on a set of polynomial features.
   4. Store the results of the quality metrics in the result dataframe.
   5. Analyze the results and select the best model according to your opinion.
   6. Additionally try different alpha parameters of regularization in algorithms, choose the best one and analyze results.

9. Native models
   1. Calculate the mean and median metrics from the previous lesson and add the results to the final dataframe.

10. Compare results
    1. Print your final tables
    2. What is the best model?
    3. Which is the most stable model?

11. Addition task
    1. There are some tricks with the target variable for better model quality. If we have a distribution with a heavy tail, you can use a monotone function to "improve" the distribution. In practice, you can use logarithmic functions. We recommend that you do this exercise and compare the results. But don't forget to do the inverse transformation if you want to compare metrics.
    2. The next trick is outliers. The angle of the linear regression line depends strongly on outliers. And often you should remove these points from !allert! only training data. You should explain why they were removed from the training sample only.  We recommend that you do this exercise and compare the results.
    3. It will also be a useful exercise to implement a linear regression algorithm with batch and mini-batch training or analytical solution (as mentioned in 4.1).

### Submission

Save your code in Python JupyterNotebook. Your peer will load it and compare it with the basic solution. Your code should include answers to all mandatory questions. The additional task is up to you. 
