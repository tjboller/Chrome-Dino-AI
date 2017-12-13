# Chrome Dinosaur Game

## Intro

This repo contains code to play the chrome dino/T-rex game.

You can play the game here http://apps.thecodepost.org/trex/trex.html

It's a super simple game, there is a T-Rex dino that stays in the same location on your screen and cacti come and you have to press space to jump over them. The longer your dino doesn't hit a cactus the higher your score. The simplicity allows the focus to be on trying different AI algorithms and comparing the performance. 

I made this code for fun and to practice OpenCV, noisy optimization, reinforcement learning. You could probably create a better AI using a more physics based solution. 

#### Dependencies:

```
pyautogui
mss
```
#### To try yourself:

You can try different AIs yourself by running the follow code at the bottom of the train_linear.py function (in the if name = main section)

To run a rule based AI:
```python
rule = 75
ai = AIs.RuleBased(75)
run.run(ai=ai)
```

To run a logistic based AI:
```python

# Equivalent to a a rule based AI with rule = 75
theta = [75, -1, 0, 0]

# Based on SPSA optimization
theta = [ 72.13388795,  -0.69299553,  -0.97952921,   0.91166942]

ai = AIs.Logistic(theta)
run.run(ai=ai)
```

To train with SPSA: 
```python

# Equivalent to a a rule based AI with rule = 75
theta_0 = [75, -1, 0, 0]

ai = AIs.Logistic(theta_0)

# will return the final theta, the test score history, and the history of all theta vectors
# ai.get_performance_cost returns the negative average of 3 runs by default
result = spsa(ai.get_performance_cost, ai.strategy, file_name='logistic', c=1, a=.5)
```

## Set Up

I use match template in OpenCV to find where in the screen the game is playing (region of interest). Then I use a python package called MSS (multiple screen shots) to quickly take a greyscale screen shot of the game region. To find the obstacles (the cacti) all we have to do is take average the column pixel values. The columns that are sufficiently dark have an obstacle. Since the cacti often occur in little groups, we have to define a minimum distance parameter such that if the number of pixels between two columns containing cacti is less than the distance parameter, it is considered the same obstacle. 

The state $s$ of the game is defined as :

\begin{equation}
s = \begin{bmatrix}
\text{Distance to the nearest obstacle} \\
\text{Width of nearest obstacle}\\
\text{Seconds since the start}
\end{bmatrix}
\end{equation}

We do not need *really* to carry information about other obstacles as the optimal jump is always to just make it over the first obstacle to give the maximum amount of time to the next obstacle. (We could learn a risk aversion type thing where if the next obstacle is far away it will jump with plenty of room on both sides, but I figure the added benefit would be small so the extra dimensionally wouldn't be worth it)

The seconds from start acts as the score and as a proxy for the speed at which the obstacles move.


## AIs 

We can think of our AI strategy as a function that takes as input the state $s$ and returns a scalar between $0$ and $1$ such that if the scalar is above $.5$ the dinosaur should jump.

The true fitness of the strategy is the _expected_ run time of the game. Obviously we cannot directly measure the expected run time, but we can get an estimate using the average of $n$ different runs.

There are two main challenges to training the AI for our simple dinosaur game.
1. The score is a noisy measurement of the true fitness of the strategy. 
2. Function evaluation is expensive. The better the strategy the longer the function runs but often a single game run takes between 5 and 100 seconds.

You can also see that there is a trade-off between the two challenges. We can reduce the noise in our measurements by increasing $n$, the number of runs averaged to get an estimate of the true fitness. As we increase $n$ the expected computational time increases linearly while the variance only decreases in proportion to the square root of $n$. 

### Rule Based

This AI just says "If the nearest obstacle is less than $x$ pixels away, jump"


##### Results
It's not hard to explore the entire reasonable parameter space for the rule $x$. The graph below shows the average score and error bars for 10 runs for each rule

<a href="https://imgur.com/qaIkRwM"><img src="https://i.imgur.com/qaIkRwM.png" /></a>


You'll see the error bars are quite large indicating that our function is pretty noisy - the same rule will get you very different scores. You'll also notice that there is not a huge difference in the average scores as the rule changes but it looks like 69-77 is best. 

To test out how SPSA (simultaneous perturbation stochastic approximation) works for our function we can implement it here and see how it does. The 'cost' of each run is implemented as the negative score for three different runs. You can see how the rule converges for an initial rule of 60 and 80 below.

<a href="https://imgur.com/TKN0CiQ"><img src="https://i.imgur.com/TKN0CiQ.png" title="source: imgur.com" /></a>

You can see that both of the optimization runs converge rather nicely at around 75, providing confidence that the SPSA algorithm worked here and that 75 is the optimum rule. (SPSA parameters: a = 5, c = 20)

Every ten updates to the rule we run three games to see how the average score changes as the rule changes. I also played the game ten times myself and took the average for comparison (disclaimer: I am probably not very good at the game)

<a href="https://imgur.com/LDDMDch"><img src="https://i.imgur.com/LDDMDch.png" title="source: imgur.com" /></a>

Not really surprised I am able to outperform the rule based system since it doesn't change as the speed of the game increases.

### Logistic Decision


##### Description 

The logistic decision algorithm learns a vector 

\begin{equation}
\theta = \begin{bmatrix}\theta_0 & \theta_1 & \theta_2 & \theta_3\end{bmatrix}
\end{equation}

The dinosaur will jump if the decision scalar $d$ is greater than $0.5$. The decision scalar is calculated as 

\begin{equation}
d = \sigma(\theta \cdot s)
\end{equation}

or

\begin{equation}
d = \sigma(\theta_0 + \theta_1 \cdot \text{Distance to the nearest obstacle} + \theta_2 \cdot \text{Width of nearest obstacle} + \theta_3 \cdot \text{Seconds since the start})
\end{equation}

where $\sigma$ is the logistic function. 


##### Results 

We use SPSA (simultaneous perturbation stochastic approximation) algorithm to optimize the $\theta$ vector to get the optimum expected score. The 'cost' of each run is implemented as the negative score for three different runs.

SPSA was chosen because it uses only function measurements, fast (requires only two objective function measurements per iteration regardless of the dimension of the optimization problem) and robust to function noise. 


Taking the SPSA parameters a = .5, c = 1 you can see how $\theta$ changes over the thousand optimization iterations. Unfortunately, the theta vector didn't converge nicely. 

<a href="https://imgur.com/cc75sTt"><img src="https://i.imgur.com/cc75sTt.png" title="source: imgur.com" /></a>

Every ten updates to $\theta$ we run a test of three runs see how the score changes over time. You can see how the scores change below.

<a href="https://imgur.com/Irji2Jq"><img src="https://i.imgur.com/Irji2Jq.png" title="source: imgur.com" /></a>

The logistic regression is getting some good results but it is very noisy and the theta vector never converged nicely (maybe my SPSA parameters were a little off, which by the way is a big disadvantage of SPSA). Hopefully we'll see more robust results with a genetic neural net approach.

