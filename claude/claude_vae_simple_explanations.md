# A short tutorial: from neural networks to the thesis VAE

This is a teaching document. It assumes you have written some Python
but not necessarily any PyTorch. It builds up, step by step, from
"what is a neural network" to "why does the exposé need a VAE in
particular, and not something simpler". Read it once front to back, then
open the scripts and read them again with the tutorial open next to
them.

The three scripts you should run alongside this document:

* `01_mlp.py` -- a multi-layer perceptron predicting tomorrow's return
* `02_simple_vae.py` -- a simple VAE producing a one-day VaR forecast
* `03_complex_vae_rolling.py` -- a larger VAE plus a 50-day rolling
  forecast loop, side-by-side with the simple VAE

---

## Table of contents

1. [What is a neural network?](#1-what-is-a-neural-network)
2. [The first script: a multi-layer perceptron (MLP)](#2-the-first-script-a-multi-layer-perceptron-mlp)
3. [Why the MLP cannot answer the thesis's question](#3-why-the-mlp-cannot-answer-the-thesiss-question)
4. [Generative models: a different way to use neural networks](#4-generative-models-a-different-way-to-use-neural-networks)
5. [The exposé's pipeline](#5-the-exposés-pipeline)
6. [The second script: a Variational Autoencoder (VAE)](#6-the-second-script-a-variational-autoencoder-vae)
7. [What goes wrong, and why the thesis exists](#7-what-goes-wrong-and-why-the-thesis-exists)
8. [Mini-batches, early stopping, and the more complex VAE](#8-mini-batches-early-stopping-and-the-more-complex-vae)
9. [The third script: the 50-day rolling forecast](#9-the-third-script-the-50-day-rolling-forecast)
10. [Where to go from here](#10-where-to-go-from-here)

---

## 1. What is a neural network?

A neural network is a function. Specifically, it is a function with a lot
of internal parameters that you can **adjust** to make the function do
what you want.

The very simplest neural network -- what we will use in `01_mlp.py` --
is built like this:

```
input  ─▶  Linear  ─▶  Nonlinearity  ─▶  Linear  ─▶  Nonlinearity  ─▶  Linear  ─▶  output
              W1, b1                          W2, b2                          W3, b3
```

Read each arrow as "apply this transformation". The two transformations
that appear are:

**Linear layer** with parameters W (a matrix) and b (a vector):

    output = W · input + b

If the input is a 5-dim vector and you want the output to be 16-dim, W
is a 16x5 matrix and b is a 16-dim vector. The numbers inside W and b
are the things the network will *learn*.

**Nonlinearity** (also called "activation function"). The standard
choice is **ReLU**:

    ReLU(x) = max(0, x)

You apply it element by element. Without these nonlinearities the
network would just be one big linear function -- it couldn't capture
any non-linear pattern in the data.

That's it. Stack `Linear → ReLU → Linear → ReLU → Linear` and you have
a "multi-layer perceptron" (MLP). The name sounds fancier than the
thing.

### How does the network learn?

You give it pairs of (input, desired output). For each pair you compute:

1. **Forward pass**: feed the input through the network, get a prediction.
2. **Loss**: compare the prediction to the desired output. The standard
   choice for regression is mean squared error: `loss = mean((y_true - y_pred)^2)`.
3. **Backward pass**: ask PyTorch "how should I adjust every parameter
   to make this loss smaller?". This step is called backpropagation.
   You don't write it yourself; PyTorch's `autograd` does it.
4. **Update**: nudge each parameter a tiny amount in the direction that
   decreases the loss. This is gradient descent.

Repeat for many (input, output) pairs and the network gets better at
the task. That's the whole training procedure.

---

## 2. The first script: a multi-layer perceptron (MLP)

Open `01_mlp.py` and look at the architecture:

```python
self.net = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)
```

Five-dim input → 16-hidden → 16-hidden → 1-dim output. The 5 input
dimensions correspond to today's returns on the 5 assets in
`qrm2025_returns.csv`. The 1 output is the network's prediction for
tomorrow's portfolio return.

The training loop is right there in the script:

```python
for epoch in range(1, 501):
    optimizer.zero_grad()              # clear stale gradients
    y_pred = model(X_train_t)          # forward pass
    train_loss = loss_fn(y_pred, y_train_t)
    train_loss.backward()              # backward pass
    optimizer.step()                   # update parameters
```

Those five lines are the heart of every PyTorch model you will ever
write. If you only memorize one thing from this tutorial, memorize
this loop.

### One subtlety: the chronological split

In financial time series you must split data in **time order**, never
shuffle. We use the first 80% (earliest in time) for training and the
last 20% for testing. If you shuffle, you accidentally let the model
"see the future" -- a bug called look-ahead bias that destroys any
honest evaluation.

### What the result looks like

When you run the script you get something like:

```
Test R^2:  +0.0086
```

R² (the coefficient of determination) measures how much variance the
model explains beyond a naive baseline of "always predict the long-run
mean". `R² ≈ 0` means the network is no better than that baseline.
This is the honest, expected result -- daily returns are very close to
unpredictable from yesterday's returns alone.

That should bother you a little: we built a neural network, trained it
for 500 epochs, and it failed. **The lesson is not that neural networks
don't work. The lesson is that we asked the wrong question.**

---

## 3. Why the MLP cannot answer the thesis's question

What does the thesis actually want to compute? Look at the exposé:

> "Forecasting portfolio risk measures such as Value-at-Risk (VaR) and
> Expected Shortfall (ES) requires an adequate model for the
> multivariate return distribution."

The thesis is not asking "what will tomorrow's return be?". It is
asking "what is the 99th percentile of tomorrow's portfolio LOSS?".

Those are very different questions. To answer the first you only need a
single number -- a point estimate of tomorrow's return. To answer the
second you need an entire DISTRIBUTION of tomorrow's returns. The 99th
percentile of a distribution is a property of the whole distribution,
not just its mean.

Here is the picture:

```
MLP world (script 01):
  input x  ─▶  network  ─▶  one number y_hat    "the most likely y"

VAE world (script 02 and the thesis):
  ???      ─▶  network  ─▶  10,000 plausible y's    "the distribution of y"
```

The MLP can only do the first. We need a generative model that can do
the second.

---

## 4. Generative models: a different way to use neural networks

A **generative model** is a neural network that, after training, can
**sample** new data points that look like the training data.

To make this concrete: imagine your training data consists of 500
five-asset return vectors (one row per trading day). A good generative
model trained on this data should be able to:

* output 10,000 brand new five-asset return vectors,
* such that the new vectors have the same per-asset mean and standard
  deviation as the real data,
* the same correlations between assets,
* the same fat-tailed behavior,
* and so on for any statistical property that mattered in training.

Once you have 10,000 such sampled vectors, computing the 99% VaR is
just sorting them and looking at the 1st-percentile portfolio return.

So **the question shifts** from "build a function that predicts the
mean" (the MLP) to "build a function that generates plausible data"
(the VAE).

### Why this is harder

Predicting the mean is one number. Producing a full distribution is
infinitely many numbers. The whole shape matters. In particular, the
TAILS of the distribution (rare extreme events) matter the most for
risk forecasting, because that's where the 99th percentile lives.

This shift -- from "predict the conditional mean" to "model the
conditional distribution" -- is the central conceptual jump from
script 01 to script 02.

---

## 5. The exposé's pipeline

Now we can read the exposé's pipeline and understand why it has the
shape it has.

The exposé's plan (from page 1, "General Framework"):

```
Step A:  daily returns ──▶  ARMA-GARCH ──▶  standardized residuals
         5 columns         (one model per      (5 columns, but now
         time series         asset, models      they look more like
                             volatility)        i.i.d. shocks)

Step B:  standardized   ──▶  DEPENDENCE  ──▶  10,000 simulated
         residuals          MODEL              standardized residuals
                            (the VAE)

Step C:  simulated      ──▶  transform   ──▶  10,000 simulated
         residuals          back to            portfolio returns
                            return scale

Step D:  10,000 sim.    ──▶  empirical    ──▶  99% VaR and 97.5% ES
         returns             quantile +
                             tail mean
```

### Why two stages? (Sklar's theorem)

Why not just dump the raw returns into the VAE? Because of a classical
result called Sklar's theorem (1959). Roughly: any multivariate
distribution can be decomposed into

1. the per-variable **marginal** distributions (one for each asset), and
2. the multivariate **dependence** between the variables.

Both pieces matter. Per-asset behavior includes things like volatility
clustering (calm weeks vs. wild weeks) -- this is best handled by a
classical, well-understood model: GARCH. Cross-asset dependence is what
makes diversification work or fail; that is the part where neural
networks might add value.

So we let GARCH do what GARCH does well (per-asset volatility), and we
hand the **standardized residuals** -- which have all the per-asset
volatility stripped out -- to the VAE. The VAE's only job is to learn
how the residuals **depend on each other across assets**.

The standardized residuals look approximately like:

* mean 0 in each column (one column per asset),
* std 1 in each column,
* still correlated across columns (positive correlation between equity
  indices, slight negative between stocks and treasury bonds, etc.).

The VAE has to capture exactly those cross-asset correlations -- and,
crucially, the joint tail behavior.

---

## 6. The second script: a Variational Autoencoder (VAE)

Open `02_simple_vae.py`. Here is the architecture in pseudo-diagram form:

```
x (5-dim)  ──▶  Linear(5, 16)  ──▶  tanh  ──▶  hidden (16-dim)
                                                     │
                                          ┌──────────┴──────────┐
                                          ▼                     ▼
                                    Linear(16, 2)         Linear(16, 2)
                                    = mu                  = log_var
                                          │                     │
                                          └──── sample z ───────┘
                                                z = mu + sigma * eps
                                                eps ~ N(0, 1)
                                                     │
                                                     ▼
                                              z (2-dim)
                                                     │
                                                     ▼
                                          Linear(2, 16)  ──▶  tanh  ──▶  Linear(16, 5)
                                                                              │
                                                                              ▼
                                                                         x_hat (5-dim)
```

So a VAE is shaped like an **autoencoder** (input → smaller latent →
back to input) BUT with two differences from a plain autoencoder:

### Difference 1: the latent is a distribution, not a point

A plain autoencoder maps each input x to a single point z in latent
space. A VAE maps each x to a small Gaussian distribution
N(mu(x), sigma(x)²) over latent space. Then, to actually feed something
to the decoder, we **sample** z from that distribution.

This sampling step needs the **reparameterization trick** to keep
gradients flowing:

    z = mu(x) + sigma(x) * eps,    eps ~ N(0, 1)

The randomness is in `eps` (external to the network). The parameters
mu(x) and sigma(x) are differentiable network outputs. Gradients of
the loss with respect to mu and sigma are well-defined.

Without this trick, the random sampling would break autograd and the
network couldn't train.

### Difference 2: the loss has a second term -- the KL divergence

The loss is

    Loss  =  Reconstruction(x, x_hat)  +  beta * KL( N(mu, sigma²) || N(0, I) )

The first term is just MSE -- the same as in the MLP. It says "the
decoded x_hat should look like the input x".

The **KL term** is new and is the magic. It compares the encoder's
output distribution N(mu(x), sigma²(x)) to the standard normal N(0, I).
The closer they are, the smaller the KL. Loosely:

    KL ≈ 0    ⟺    the encoder maps every x to roughly N(0, I)
    KL >> 0   ⟺    the encoder produces something wildly non-normal

We MINIMIZE the loss, so the network is pushed toward "encoder output
looks like N(0, I)".

### Why this is enough to make the model generative

After training, the encoder has been pushed (by the KL term) to map
real data to a region of latent space that overlaps with N(0, I). To
generate new data, we simply:

1. Sample `z ~ N(0, I)` (the prior),
2. Pass z through the decoder,
3. Get a synthetic x.

Because real data was mapped INTO the N(0, I) region, and the decoder
learned to reconstruct real data from points in that region, drawing
fresh N(0, I) samples and decoding them produces **outputs that look
like the training data**.

That is the generative recipe. Five lines of code in script 02
(`@torch.no_grad() def sample(self, n): ...`).

### The closed-form KL for diagonal Gaussians

You won't need to derive it, but for reference:

    KL[ N(mu, diag(sigma²)) || N(0, I) ]
        = 0.5 * sum( mu² + sigma² - log(sigma²) - 1 )

This is what the script's `vae_loss` function computes in one line. The
network outputs `log_var = log(sigma²)` directly (instead of `sigma`),
because `log_var` can be any real number while `sigma` has to be
positive -- and unconstrained outputs are easier for neural networks.

### Beta

The coefficient `beta` weights the KL term. `beta = 1` is the textbook
VAE. The exposé's baseline uses `beta = 2`. Higher beta means more
pressure on the latent to look Gaussian, at the cost of worse
reconstruction.

Why does the exposé prefer `beta = 2` instead of `beta = 1`? It is a
deliberate tradeoff: a more strongly-regularized latent makes the
generative sampling step more reliable but produces blurrier samples.
The exposé's **L2 sensitivity experiment** asks exactly this question
-- whether beta=1 or beta=2 or beta=4 produces lower model risk in
practice. We use beta=1 here for clarity. The thesis answers the question.

---

## 7. What goes wrong, and why the thesis exists

When you run `02_simple_vae.py`, look at the diagnostics block. You will
see something like:

```
empirical std:      [1.01 1.01 1.00 1.01 0.99]
simulated std:      [0.71 0.68 0.71 0.68 0.42]   <-- watch this

Tail-coverage check (target: ~1.0% in each tail):
   STOXX_EU_600:           below 1%: 0.00%   above 99%: 0.00%
   DOWJONES_INDUSTRIALS:   below 1%: 0.00%   above 99%: 0.00%
   ...
```

Two problems jump out:

### Problem 1: variance shrinkage

The simulated residuals have standard deviation around 0.7, when the
real residuals have standard deviation 1.0. The VAE is producing
samples that are systematically less spread out than the training data.

**Why?** MSE-trained VAEs have an averaging bias. To minimize MSE the
decoder is pulled toward producing the *expected* x for any latent z.
The expected value of a noisy quantity is less variable than any
individual realization. So the simulated outputs sit too close to the
mean.

### Problem 2: zero tail coverage

The "tail coverage" check counts how often a simulated sample lands
in the most extreme 1% of the training data. For a perfect generative
model this would be exactly 1%. Our simple VAE gives **0%** -- it
never produces an extreme observation.

This is **catastrophic for risk forecasting**. The whole point of VaR
is the tail. If the VAE can never reach the empirical 99th percentile,
the VaR forecast it produces will be too small. You will hold too
little regulatory capital. You will be confident exactly when you
should be most cautious.

### Why this is the thesis's research question

The exposé asks: "Can a VAE serve as a viable, data-driven alternative
to parametric copulas for modeling multivariate return dependencies?".
The honest answer from script 02 is: **not in the simplest form**. The
variance shrinkage is severe and the tail coverage is zero.

The thesis's job is to find out:

* Can we fix this with a different architecture (latent dim, hidden
  width)? -- exposé experiments L1, L4.
* Can we fix this by tuning beta? -- experiment L2.
* Can we fix this by changing the loss function to specifically reward
  matching the tails? -- experiment L3.
* If we do fix it, how does the resulting model risk compare to the
  copula benchmark from Fritzsch et al. (2024)?

You now know enough to understand why every one of those questions is
the right one to ask. They are all aimed at the two problems you just
saw on screen.

---

## 8. Mini-batches, early stopping, and the more complex VAE

Script 02 trained on the full dataset in one shot, used a small network,
and stopped after a fixed 500 epochs. That is the simplest setup but it
leaves three pieces of standard neural-network practice on the table.
Script 03 adds them. Let's walk through each.

### 8.1 Mini-batches

In script 02 we computed the loss on ALL training rows simultaneously
and took ONE gradient step per epoch. That is called **full-batch
gradient descent**.

The more standard approach is **mini-batch gradient descent**: split
the training set into chunks of, say, 32 rows, compute the loss and
gradient on each chunk in turn, and take a gradient step after each
chunk. Over one epoch the network sees the same data but takes many
small steps instead of one big one.

Why this is better in practice:

* **More gradient steps per epoch** -> faster progress in wall-clock time.
* **Noisier gradient estimates** -> the noise acts as a regularizer
  (it stops the optimizer from settling deep into bad local minima).
* **Bounded memory** -> matters once your data is too big to fit in
  memory at once. (Not a concern at our scale, but a habit worth
  building.)

The user instruction for this script is **batch size 32**, which is a
standard small-data choice. Script 03's `train_vae` function uses
PyTorch's `DataLoader` to do the shuffling and chunking:

```python
loader = DataLoader(TensorDataset(Z_tr), batch_size=32, shuffle=True)
for (xb,) in loader:
    optimizer.zero_grad()
    x_hat, mu, lv = model(xb)
    vae_loss(xb, x_hat, mu, lv, beta=model.beta).backward()
    optimizer.step()
```

Two subtleties worth flagging:

1. We `shuffle=True` -- but only WITHIN the training portion. We split
   the data chronologically BEFORE creating the loader. That way the
   validation set is still "the future" relative to training.

2. The "step every batch" pattern uses 32 samples per gradient. With
   ~420 training samples that means ~13 gradient steps per epoch.
   Compared to script 02 (one step per epoch) we are taking 13x more
   updates per pass through the data.

### 8.2 Early stopping

Script 02 trained for exactly 500 epochs. That is wasteful in both
directions:

* If the model fully fits in 50 epochs, the remaining 450 epochs are
  doing nothing useful (or worse, they are slowly overfitting).
* If the model needs more than 500 epochs, we cut training off too soon.

**Early stopping** fixes this. We hold out the last 15% of the training
window as a "validation set" and monitor the loss on it after every
epoch. If the validation loss has not improved for `patience` (here:
10) epochs, we stop and roll the model back to its best-validation
checkpoint.

This is the single best regularization technique you can use with
almost no additional cost. It is built into script 03's `train_vae`:

```python
if val < best_val - 1e-7:
    best_val   = val
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
else:
    bad += 1
    if bad >= patience:
        break

model.load_state_dict(best_state)
```

### 8.3 The "more complex" VAE

The script's `ComplexVAE` is the same idea as `SimpleVAE` from script
02, scaled up:

|                 | SimpleVAE (script 02)        | ComplexVAE (script 03)             |
|-----------------|------------------------------|------------------------------------|
| Hidden layers   | 1 layer x 16 units           | 2 layers x 64 units                |
| Latent dim      | 2                            | 3                                  |
| beta            | 1.0                          | 2.0                                |
| Activation      | tanh                         | tanh                               |
| Trainable params| ~400                         | ~9,000                             |
| Training        | full-batch, 500 epochs       | batch 32, early stopping           |

The shape is exactly the exposé's "Baseline configuration" (page 2):
two hidden layers x 64 units, tanh, latent_dim = 3, beta = 2. We have
brought you here on purpose. Once you understand `ComplexVAE`, you
understand the entire architectural starting point of the thesis.

Two natural questions:

**Why is ComplexVAE "more complex"?** Two reasons. The extra hidden
layer increases the network's *capacity* (the family of functions it
can represent). The larger latent dimension gives the model three
independent directions of variation instead of two.

**Why beta = 2 instead of beta = 1?** It is a deliberate choice. A
higher beta puts more pressure on the encoder to produce N(0, I) -shaped
distributions in latent space. The generative sampling step at
inference time (draw z from N(0, I), decode) is then more reliable --
fewer "off-manifold" samples that the decoder has never seen. The price:
the network is given LESS freedom to encode fine-grained information
about x, which usually worsens reconstruction quality and tail
coverage. **This trade-off is exactly the L2 sensitivity experiment in
the exposé.** You will see the consequences in the next section.

---

## 9. The third script: the 50-day rolling forecast

Open `03_complex_vae_rolling.py`. The key change from script 02 is
that we no longer produce ONE VaR forecast on ONE window. We produce
50 VaR forecasts on 50 windows -- a time series.

### 9.1 Why this matters

A single forecast is a snapshot. It tells you the number on one day,
but it cannot tell you:

* Whether the model's forecasts are STABLE day-to-day or whether they
  swing around as new data arrives.
* Whether the model's BIAS is consistent (always too high? always too
  low?) or noisy.
* Whether two competing models drift apart over time, or whether they
  disagree by a steady amount.

These are all properties of the time series, not of any single
forecast. The thesis quantifies them on the FULL sample of 2,000+
days; this script does the smallest legible version: 50 days.

### 9.2 The rolling-window mechanics

The picture for ONE day `t` in our 50-day window:

```
   in-sample window of 500 days    forecast for day t (one step ahead)
   |<---------------------->|     |
   day t-500 ... day t-1     day t
   |                          |
   |__ fit GARCH marginals    |
   |__ train the VAEs (every  |
        10 days only)         |
                              |__ sample 10k synthetic residuals from each VAE
                              |__ transform to returns using GARCH's
                                  one-step-ahead mu_{t} and sigma_{t}
                              |__ compute portfolio quantile -> VaR_{t}
```

We slide this window forward by one day, 50 times. The model is NEVER
allowed to see day `t` while training -- that would be look-ahead bias.

### 9.3 Two computational shortcuts and what they mean

A purist version of the thesis pipeline would retrain everything every
day. The script makes two compromises to keep runtime under 30 seconds:

1. **GARCH refits every day.** Good. Matches the exposé exactly.

2. **VAEs refit every 10 days only.** Between refits, we keep the
   previous VAE's weights and just feed it the new GARCH outputs.
   The exposé's L5 experiment varies this between daily, quarterly,
   and annual refits -- so this shortcut is itself a thesis-relevant
   knob, not a hidden assumption.

When you look at the plot you will see the consequences: between
refits the VaR forecasts move smoothly (only the GARCH inputs change);
on refit days there is a small jump (the VAE weights change).

### 9.4 What you should observe in the output

Three things deserve attention.

**Observation 1: the two models track each other but at different
levels.** ComplexVAE consistently sits below SimpleVAE -- by about
0.1-0.2 percentage points of VaR. This is the variance-shrinkage
problem from section 7 made concrete: the higher-beta model produces
even more conservative-looking samples, and so produces a SMALLER VaR.

If you look at the plot, you will see the gap is not random noise --
it is a structural difference between the two configurations, present
on essentially every day.

**Observation 2: model risk is a real, measurable quantity.** The
script prints "mean MAD over 50 days". With two models, MAD is just
half the absolute distance between their forecasts. On the data the
script outputs roughly:

```
mean MAD over 50 days:  0.10%   (= ~$340 on a $100,000 portfolio, 10-day scaled)
max  MAD over 50 days:  0.18%
```

For comparison: Fritzsch et al. (2024), with **180** competing
Copula-GARCH models, report an average MAD around 0.165%. So even our
two-model toy is already producing model risk of the same order of
magnitude as the published paper.

**Observation 3 (the punchline): the VaRs are SEVERELY MISCALIBRATED.**
At the end of the script you will see a "realized < -VaR violations"
line. For a 99% VaR over 50 days, you would expect on average 0.5
violations (and would not be surprised by 0 or 1 or 2). What you
actually see:

```
SimpleVAE     ~ 10 violation(s) out of 50 days
ComplexVAE    ~ 13 violation(s) out of 50 days
```

That is **TWENTY TIMES** the expected violation rate. The VaR is far
too small -- meaning the model badly underestimates the size of typical
losses.

This is the variance-shrinkage problem from section 7 expressed in its
most consequential form. A risk manager using this model would hold
**far too little capital**, would be violating regulatory backtests
constantly, and would be exposed to losses they had been told (by the
model) were near-impossible.

This is precisely the gap the thesis exists to investigate. The exposé's
L1-L7 sensitivity experiments are aimed at finding VAE configurations
where this miscalibration is manageable. Right now, with the simplest
training set-up, it is severe.

### 9.5 What 50 days does and doesn't tell you

Fifty days is a small sample. The numbers in the "violation count"
line are honestly very noisy -- the 95% confidence interval on
"violations from a 99% VaR over 50 days" is approximately {0, ..., 4}
even if the model is perfectly calibrated. Our observed 10-13 is
unambiguously outside that band, but a more precise statement would
require many more days. The full thesis runs on 1,800+ days; this
script just demonstrates the structure.

Fifty days IS enough to:

* See the time-series shape of each model's forecasts.
* Compare the levels and the day-to-day variability of two competing
  models.
* Quantify their mean disagreement.
* See the refit jumps and understand why the exposé's L5 sensitivity
  experiment matters.

---

## 10. Where to go from here

The right order to take next:

1. **Re-read all three scripts** with the tutorial open. Now the
   reparameterization trick, the KL term, the chronological split, the
   batch-vs-full-batch question, and early stopping should all make
   sense.

2. **Modify hyperparameters in script 03** and re-run.
   * Change ComplexVAE's `beta` from 2.0 to 1.0 -- does the violation
     rate get better or worse? Why?
   * Change `latent` from 3 to 5 -- what happens to the gap between
     the two models?
   * Change `REFIT_EVERY` from 10 to 5 to 1 -- does the time series
     get smoother or rougher?
   * Each of these IS one of the L1-L7 experiments from the exposé.
     You are now doing the thesis's sensitivity study by hand on a
     50-day window.

3. **Look at the plot `rolling_var.png`**. The upper panel shows the
   two VaR series. The lower panel shows their daily MAD. This is the
   exact analogue of Figure 1 in Fritzsch et al. (2024), at a much
   smaller scale.

4. **Open the thesis prototype** (the `thesis/` folder from your
   earlier work). Every file there has a direct analogue in this
   intro: `02_simple_vae.py` corresponds to `thesis/src/models.py`;
   the GARCH block here is exactly `thesis/src/marginals.py`; the
   sampling-and-quantile block is `thesis/src/forecasting.py`. The
   thesis just wraps everything in a rolling window of ~2,000 days
   (vs. our 50), adds the copula benchmarks, and runs the formal
   backtests.

---

## Glossary of key terms

* **MLP (multi-layer perceptron)**: a feed-forward neural network made
  of `Linear → activation → Linear → activation → ... → Linear` layers.
  No autoencoder structure, no latent space.

* **Autoencoder**: a neural network shaped like `input → bottleneck →
  output`, trained to make the output match the input. The bottleneck
  forces a compressed representation.

* **VAE (Variational Autoencoder)**: an autoencoder with a
  probabilistic bottleneck. The encoder outputs a distribution over
  the bottleneck, not a point. Includes a KL term in the loss that
  makes the model generative.

* **Latent space**: the bottleneck. The compressed representation
  produced by the encoder. In our script its dimension is 2.

* **Reparameterization trick**: writing `z = mu + sigma * eps` with
  `eps ~ N(0, I)`. Lets gradients flow through the sampling step.

* **KL divergence**: a measure of how different two probability
  distributions are. In a VAE we use it to keep the encoder's output
  distribution close to a standard normal prior.

* **Beta**: the weight on the KL term in the VAE loss. `beta = 1` is
  the textbook value; the exposé's baseline uses `beta = 2`.

* **Standardized residual**: what comes out of a GARCH model after you
  subtract the conditional mean and divide by the conditional standard
  deviation. Should look like zero-mean, unit-variance noise.

* **Sklar's theorem**: the result that justifies modeling marginals
  and dependence separately. Every multivariate distribution factors
  into per-variable marginals plus a copula (or any other dependence
  model, like a VAE).

* **VaR (Value-at-Risk)**: the alpha-th percentile of the loss
  distribution. "99% VaR = 2.0%" means "with 99% probability tomorrow's
  loss will not exceed 2.0% of the portfolio value".

* **ES (Expected Shortfall)**: the average loss given that loss exceeds
  VaR. Captures the size of losses in the worst alpha% of cases.

* **Variance shrinkage**: the tendency of MSE-trained VAEs to produce
  samples that are less variable than the training data. Causes
  underestimation of risk measures.

* **Mini-batch**: a small chunk of the training data (here: 32 rows).
  Mini-batch gradient descent computes the loss and the gradient on
  one mini-batch at a time, taking many small steps per epoch rather
  than one large one.

* **Epoch**: one full pass through the training data. With a batch
  size of 32 and 420 training rows, one epoch is about 13 gradient
  steps.

* **Early stopping**: train until the validation loss stops improving
  for a fixed number of epochs (the "patience"), then revert to the
  best-validation checkpoint. Cheap, very effective regularization.

* **Validation set**: a chronologically-latest chunk of the training
  window (here: the most recent 15%) that the model is NOT trained on.
  Used to detect when training has begun to overfit.

* **Rolling window**: a fixed-length training window that slides
  forward in time. On each new day, we drop the oldest observation
  and add the newest, refit the model, and produce a one-step-ahead
  forecast. This is the standard forecasting protocol in finance.

* **One-step-ahead forecast**: a forecast for time t+1 made using
  data up to and including time t -- never beyond. Synonymous with
  "out-of-sample" forecast.

* **Look-ahead bias**: accidentally letting future data influence the
  training of a model that will be evaluated as if it had only seen
  past data. A silent bug that makes models look much better than
  they are. Avoided by chronological splits and never shuffling
  before splitting.

* **Backtest violation**: a day on which the realized loss exceeded
  the VaR forecast. For a calibrated 99% VaR, violations should
  occur on about 1% of days. Far more than that is evidence of an
  under-conservative model.
