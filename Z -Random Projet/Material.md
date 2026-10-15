# Spiking Neural Networks with STDP for Simulated Robotics

A hands-on course: LIF neurons, excitatory and inhibitory plasticity implemented with synaptic traces, reward-modulated learning in a closed sensorimotor loop, and how to make a NumPy implementation fast.

Everything here was run and checked. The complete scripts are in the appendix: `snn_robot.py` (network + robot, trains a light-seeking vehicle in about 10 s on one CPU core) and `bench.py` (the optimisation benchmark).

**Prerequisites:** NumPy, basic ODEs, a little probability. No neuroscience background assumed.

**Course map**

1. The leaky integrate-and-fire neuron
2. Synapses and the E/I architecture
3. STDP and the trace trick
4. Excitatory plasticity in practice (stability and homeostasis)
5. Inhibitory plasticity (Vogels–Sprekeler iSTDP)
6. Closing the loop: reward-modulated STDP on a simulated robot
7. Optimising the NumPy implementation
8. Exercises
9. References and appendix code

---

## Module 1 — The leaky integrate-and-fire neuron

The membrane potential relaxes toward rest and is pushed by synaptic input:

$$\tau_m \frac{dv}{dt} = (v_{rest} - v) + g_e(t) - g_i(t)$$

When $v \ge v_{th}$ the neuron spikes, $v$ is reset to $v_{reset}$, and it is clamped for a refractory period $t_{ref}$. Here $g_e, g_i$ are expressed in mV (a "current-based" synapse: input shifts the equilibrium voltage). A conductance-based version multiplies by driving forces, $g_e(E_e - v)$, which is more realistic and self-limiting, but current-based is easier to reason about and to tune.

### Exact integration instead of Euler

If the input is held constant over one step $\Delta t$, the ODE has a closed-form solution. Define the instantaneous equilibrium $v_\infty = v_{rest} + g_e - g_i$; then

$$v(t+\Delta t) = v_\infty + (v(t) - v_\infty), e^{-\Delta t/\tau_m}$$

This is unconditionally stable (forward Euler blows up when $\Delta t$ approaches $\tau$), and the factor $e^{-\Delta t/\tau_m}$ is a constant you compute once. The same applies to every exponentially decaying variable in this course: synaptic conductances, STDP traces, eligibility traces. **Each one becomes a single multiply per step.**

```python
def lif(v, ge, gi, ref, decay, t_ref, bias=0.0):
    v_inf = v_rest + bias + ge - gi
    v[:] = v_inf + (v - v_inf) * decay     # decay = exp(-dt/tau_m), precomputed
    ref -= dt
    v[ref > 0] = v_reset                   # refractory clamp
    spk = v >= v_th
    v[spk] = v_reset; ref[spk] = t_ref
    return np.flatnonzero(spk)             # spike *indices*, not a boolean mask
```

Returning indices rather than a boolean vector is deliberate: every later operation is driven by "which neurons spiked", and activity is sparse (a 20 Hz neuron spikes in 2% of 1 ms steps).

**Typical parameters used here:** $\Delta t = 1$ ms, $v_{rest} = v_{reset} = -65$ mV, $v_{th} = -52$ mV, $\tau_m = 20$ ms (E) / 10 ms (I), $t_{ref} = 2$ ms (E) / 1 ms (I).

---

## Module 2 — Synapses and the E/I architecture

### Exponential synapses

Each presynaptic spike adds its weight to the postsynaptic conductance, which then decays:

$$g_e \leftarrow g_e, e^{-\Delta t/\tau_{ge}} \quad\text{every step};\qquad g_e \leftarrow g_e + w_{ij}\quad\text{on a spike of } j$$

with $\tau_{ge} \approx 5$ ms and $\tau_{gi} \approx 10$ ms (inhibition is usually slower).

### Dale's law and the four projections

Neurons are either excitatory (E) or inhibitory (I): all outgoing weights of a neuron have the same sign. That gives four recurrent blocks plus input:

| Projection | Role                    | Plastic here?                     |
| ---------- | ----------------------- | --------------------------------- |
| Input → E  | sensory drive           | yes: STDP / reward-modulated STDP |
| E → E      | recurrence, assemblies  | yes: STDP (optional)              |
| E → I      | recruits inhibition     | fixed                             |
| I → E      | balance, rate control   | yes: inhibitory STDP              |
| I → I      | stabilises I population | fixed                             |

A common ratio is 80% E / 20% I with sparse random connectivity ($p \approx 0.1$–$0.4$). Store a boolean mask for each sparse block so that plasticity cannot create synapses that don't exist.

### Storage convention: `W[pre, post]`

With this layout, delivering the spikes of a set of presynaptic neurons is a sum over rows:

```python
geE += W_inE[in_idx].sum(0) + W_EE[e_idx].sum(0)
giE += W_IE[i_idx].sum(0)
```

That costs $O(k \cdot N_{post})$ for $k$ spikes instead of the $O(N_{pre} N_{post})$ of `s @ W`. Rows are contiguous in memory (C order), so this is cache-friendly. We'll revisit the trade-off in Module 7: potentiation touches columns, which are strided.

---

## Module 3 — STDP and the trace trick

### The pair-based rule

For a pre spike at $t_{pre}$ and post spike at $t_{post}$, with $\Delta t = t_{post} - t_{pre}$:

$$\Delta w = \begin{cases} +A_+ , e^{-\Delta t/\tau_+} & \Delta t > 0 ;;\text{(pre before post: potentiate)}\\ -A_- , e^{\Delta t/\tau_-} & \Delta t < 0 ;;\text{(post before pre: depress)}\end{cases}$$

Summing this over all spike pairs naively is $O(\text{spikes}^2)$ per synapse and requires storing spike histories.

### Traces make it local and $O(1)$ per event

Give every neuron a trace that jumps on each spike and decays exponentially:

$$x_{pre} \leftarrow x_{pre}, e^{-\Delta t/\tau_+},\quad x_{pre} \mathrel{+}= 1 \text{ on pre spike}$$ $$x_{post} \leftarrow x_{post}, e^{-\Delta t/\tau_-},\quad x_{post} \mathrel{+}= 1 \text{ on post spike}$$

At the moment the postsynaptic neuron fires, $x_{pre}(t) = \sum_{t_{pre} < t} e^{-(t - t_{pre})/\tau_+}$, which is exactly the sum of the potentiation window over every earlier pre spike. So the whole rule becomes:

- **on a post spike:** $w_{ij} \mathrel{+}= A_+, x_{pre,j}$
- **on a pre spike:** $w_{ij} \mathrel{-}= A_-, x_{post,i}$

The traces are per neuron ($O(N)$ memory), not per synapse. That is the core optimisation that makes STDP cheap.

### Design choices that matter

**All-to-all vs nearest-spike.** `x += 1` sums over all previous spikes (all-to-all). `x = 1` keeps only the most recent spike (nearest-neighbour). Nearest-spike saturates at high rates, which changes the rate dependence of learning; all-to-all is the default.

**Additive vs soft bounds.**

|                       | Potentiation             | Depression        | Behaviour                                                             |
| --------------------- | ------------------------ | ----------------- | --------------------------------------------------------------------- |
| Additive + clip       | $A_+ x_{pre}$            | $A_- x_{post}$    | Bimodal weights (0 or $w_{max}$), strong competition, can be unstable |
| Multiplicative (soft) | $A_+ x_{pre}(w_{max}-w)$ | $A_- x_{post}, w$ | Unimodal weights, stable, weaker competition                          |

This course uses soft bounds for excitatory weights.

**The balance $A_+\tau_+$ vs $A_-\tau_-$.** For uncorrelated Poisson pre and post neurons at rates $\rho_{pre}, \rho_{post}$, the expected drift of an additive rule is

$$\left\langle \frac{dw}{dt}\right\rangle = \rho_{pre},\rho_{post},(A_+\tau_+ - A_-\tau_-)$$

If $A_-\tau_- > A_+\tau_+$, uncorrelated activity weakens synapses and only causal (pre-predicts-post) correlations strengthen them. That is the classic stable setting for unsupervised learning. If the balance tips positive, the rule becomes partly a Hebbian rate rule. Module 6 shows why that is actually useful for reward-modulated learning.

**Ordering inside one time step.** Pre and post can spike in the same step. Pick a convention and keep it: here, depression is applied first using the post trace from before this step's post spikes, then traces are incremented, then potentiation uses the updated pre trace. Simultaneous spikes therefore count as potentiation.

### Code: vectorised, event-driven

```python
def exc_stdp(W, x_pre, x_post, pre_idx, post_idx, A_plus, A_minus, wmax):
    # (a) pre spikes -> depression of their ROWS (soft bound: * w)
    W[pre_idx] -= A_minus * x_post[None, :] * W[pre_idx]
    # (b) traces (all-to-all)
    x_pre[pre_idx] += 1
    x_post[post_idx] += 1
    # (c) post spikes -> potentiation of their COLUMNS (soft bound: * (wmax - w))
    W[:, post_idx] += A_plus * x_pre[:, None] * (wmax - W[:, post_idx])
```

Traces decay once per step, at the top of the loop, together with all other decaying state.

### Beyond pairs: triplet STDP

Pair STDP cannot reproduce the frequency dependence seen in cortex (high-rate pairings potentiate regardless of order). The triplet rule of Pfister & Gerstner (2006) adds a second, slower post trace $y_2$ (and optionally pre trace $x_2$): on a post spike, $\Delta w = x_{pre}(A_2^+ + A_3^+ y_2)$, where $y_2$ is read *before* its own increment. Cost: one more trace per neuron. It slots straight into the code above.

---

## Module 4 — Excitatory plasticity in practice

Plain STDP on recurrent E→E weights is a positive feedback loop: stronger weights → more correlated firing → more potentiation. Practical networks add at least one homeostatic mechanism.

**Synaptic normalisation.** After each update, rescale each neuron's incoming weights so they sum to a fixed total. Only the *distribution* over inputs is learned, never the total drive. Cost: one column sum per update.

```python
W *= target_sum / (W.sum(0, keepdims=True) + 1e-9)
```

**Adaptive threshold.** Each E neuron gets $\theta_i$ added to its threshold; $\theta_i$ jumps by a small amount on each spike and decays slowly ($\tau_\theta \sim 10^4$–$10^7$ ms). Neurons that fire a lot become harder to drive (Diehl & Cook, 2015).

**Inhibitory plasticity** (next module) controls rates through the I→E weights, leaving excitatory weights free to encode structure.

**E→E masks.** Multiply every E→E update by the connectivity mask so absent synapses stay absent, and set the diagonal to False (no autapses):

```python
W_EE[e_idx] += dep * M_EE[e_idx]
W_EE[:, e_idx] += pot * M_EE[:, e_idx]
```

---

## Module 5 — Inhibitory plasticity (Vogels–Sprekeler iSTDP)

### The rule

Vogels et al. (2011) proposed a symmetric rule for I→E synapses. With traces $x_I$ (on inhibitory pre neurons) and $x_E$ (on excitatory post neurons), both with time constant $\tau \approx 20$ ms:

- **on an inhibitory (pre) spike:** $w \mathrel{+}= \eta,(x_E - \alpha)$
- **on an excitatory (post) spike:** $w \mathrel{+}= \eta, x_I$

Near-coincident spikes in either order strengthen inhibition; every inhibitory spike also pays a constant depression $\eta\alpha$.

### Why it sets the firing rate

For uncorrelated activity, each trace has mean $\tau\rho$, so the expected drift is

$$\left\langle \frac{dw}{dt}\right\rangle = \eta,\rho_I,(2\tau\rho_E - \alpha)$$

The drift is zero exactly when $\rho_E = \alpha / (2\tau)$. Choosing

$$\alpha = 2,\rho_0,\tau$$

makes $\rho_0$ the target rate of every excitatory neuron. If a neuron fires faster, its inhibition grows; if it fires slower, inhibition shrinks. Because the rule is also sensitive to correlations, inhibition tends to become *co-tuned* with excitation (detailed E/I balance), not just a global brake.

### Code

```python
alpha = 2 * rho0 * tau_istdp          # rho0 in spikes/ms, e.g. 0.02 = 20 Hz

def inh_stdp(W_IE, M_IE, xI, xE, i_idx, e_idx, eta, alpha, wmax):
    xI[i_idx] += 1; xE[e_idx] += 1
    if i_idx.size:   # pre (I) spikes -> rows
        W_IE[i_idx] += eta * (xE[None, :] - alpha) * M_IE[i_idx]
    if e_idx.size:   # post (E) spikes -> columns
        W_IE[:, e_idx] += eta * xI[:, None] * M_IE[:, e_idx]
    np.clip(W_IE, 0, wmax, out=W_IE)
```

### Pitfalls found while building the robot

- **Inhibition can only weaken when I neurons fire.** If the whole network falls silent, the rule freezes and the network stays silent. Give E neurons a tonic bias or keep a floor on input rates.
- **I neurons must be able to fire.** If $w_{EI}$ is too weak, the I population barely spikes, I→E weights pile up at $w_{max}$, and the E rate never reaches $\rho_0$. In testing, raising $w_{EI}$ from 2 to 4 mV and $w_{max}^{IE}$ from 20 to 50 brought E rates from about 50 Hz down to the 20 Hz target.
- **A fast iSTDP fights your signal.** In a controller, what matters is the *difference* between motor populations. With a large $\eta$, iSTDP equalises every neuron's rate on the timescale of behaviour and erases that difference. Keep $\eta$ small enough that homeostasis is slower than the task.

---

## Module 6 — Closing the loop: reward-modulated STDP on a robot

Pure STDP learns correlations in its input; it does not know what the robot should *do*. For behaviour we use a **three-factor rule**: STDP proposes changes, a global neuromodulatory signal (think dopamine) decides whether to keep them.

### The three-factor rule

Instead of writing the STDP update into $W$, write it into a per-synapse eligibility trace $e_{ij}$ that decays with $\tau_e$:

$$e_{ij} \leftarrow e_{ij},e^{-\Delta t/\tau_e} + \text{STDP}*{ij}(t), \qquad \Delta W*{ij} = \eta_R,(R - \bar R), e_{ij}$$

$\bar R$ is a running average of reward; subtracting it makes the modulator zero-mean, which is critical, otherwise every active synapse drifts in the sign of the average reward. In code this is a one-line switch: the same STDP increments go into `elig` instead of `W`, and a reward step applies them.

```python
target = elig if reward_mode else W_in
target[in_idx] += dep_in
target[:, e_idx] += pot_in

def apply_reward(W, elig, delta_r, lr, wmax, target_sum):
    W += lr * delta_r * elig
    np.clip(W, 0, wmax, out=W)
    W *= target_sum / (W.sum(0, keepdims=True) + 1e-9)   # synaptic normalisation
```

### The task: a Braitenberg light-seeker

A differential-drive robot (wheel base 0.5 m) moves in a plane toward a light; when it gets within 1 m, the light jumps somewhere new. Two light sensors point ±45° from the heading and report $\tfrac12(1+\cos(\text{bearing} \mp 45°))$.

The correct solution is Braitenberg's vehicle 2b: **crossed** wiring. The left sensor should drive the right wheel so the robot turns toward the light. The network has to discover this from reward alone.

**Encoding.** Each sensor drives 20 Poisson input neurons at $10 + 80,s$ Hz. The 10 Hz floor matters (see lessons below).

**Network.** 40 inputs → 80 E + 20 I. E neurons 0–39 are the left-motor pool, 40–79 the right-motor pool. Input→E uses reward-modulated STDP, I→E uses iSTDP, E→E is fixed here.

**Control loop.** Every 20 ms (20 SNN steps): read sensors, generate input spikes, run the network, count motor spikes, then

$$\omega\text{-command: } \text{turn} = k,(r_R - r_L), \qquad v_L = v_0 - \text{turn}, \quad v_R = v_0 + \text{turn}$$

with $v_0 = 0.3$ m/s and $k = 0.05$ (m/s per Hz). This push-pull decoding uses only the rate *difference*.

**Reward.** $R_t = |b_{t-1}| - |b_t|$: how much the absolute bearing error shrank this control step.

```python
for k in range(n_ctrl):
    s = bot.sensors()
    rates = np.repeat(0.01 + s * r_max, n_per_sensor)          # spikes/ms
    in_spk = rng.random((T_ctrl, rates.size)) < rates * dt       # whole block at once
    cnt = np.zeros(n_e)
    for t in range(T_ctrl):
        e_idx, _ = net.step(np.flatnonzero(in_spk[t]))
        cnt[e_idx] += 1
    rL = cnt[:n_motor].mean() / T_ctrl * 1000
    rR = cnt[n_motor:].mean() / T_ctrl * 1000
    turn = gain * (rR - rL)
    bot.move(v0 - turn, v0 + turn, T_ctrl / 1000)
    err = abs(bot.bearing())
    r = err_prev - err; err_prev = err
    net.apply_reward(r - r_bar)
    r_bar += 0.05 * (r - r_bar)
```

### Results

After 6,000 control steps (2 simulated minutes), learning was frozen and each network was tested on a fresh 30-second episode. The metric is mean $\cos(\text{bearing})$: +1 means always facing the light, 0 is chance/orbiting, −1 means facing away.

| Seed | Untrained | Trained |
| --- | --- | --- |
| 0 | −0.80 | +0.97 |
| 1 | −0.37 | +0.62 |
| 2 | −0.10 | +0.54 |
| 3 | −0.23 | +0.29 |
| 4 | −0.28 | +0.82 |

All five seeds improved, and in every case the learned weights moved toward crossed wiring. Wall time is about 10 s per training run on a single core. For comparison, hand-set crossed weights score +0.94 and hand-set uncrossed weights −1.00, so the learned controllers reach close to the hand-designed optimum on the better seeds.

### Lessons from making it work (the part papers leave out)

The first version did not learn at all. Each fix below addressed a specific, diagnosed failure. They generalise to most R-STDP robotics setups.

1. **Silent-network trap.** With sensors reading zero when the light was behind, the network went silent, the robot drove straight, and nothing could learn (no spikes means no eligibility, and iSTDP freezes). Fix: sensor floor + tonic bias on E neurons.
2. **Flat reward landscape.** Rewarding $\Delta\cos(b)$ gives zero gradient when facing directly away, so "facing away" became a stable trap. $\Delta|b|$ has a constant gradient everywhere.
3. **Common-mode drift.** Without normalisation, one motor pool's total input kept growing and the robot learned to spin in circles. Per-neuron synaptic normalisation leaves only the useful degree of freedom: how each motor neuron splits its input between left and right sensors. This single change made learning go consistently in the right direction.
4. **Eligibility too long dilutes credit.** $\tau_e = 300$ ms spans 15 control steps, so each reward is blamed on many unrelated actions. $\tau_e = 40$ ms (about two control steps) worked far better. Set $\tau_e$ from the actual delay between action and reward.
5. **The STDP window sign flips the learning.** With the "stable" setting $A_-\tau_- > A_+\tau_+$, rate-driven eligibility is *negative*, so reward pushes weights the wrong way. For the eligibility we use $\tau_+=\tau_-=20$ ms and $A_- = 0.8,A_+$, making eligibility mildly Hebbian (positive for co-active pairs).
6. **Orbiting.** At 1 m/s, moderately good wiring makes the robot circle the light at a fixed radius ($\cos b \approx 0$) with almost no reward gradient. Reducing $v_0$ to 0.3 m/s removed the orbit.
7. **iSTDP fighting the controller.** See Module 5: slow homeostasis ($\eta = 0.005$) with a 20 Hz target.

---

## Module 7 — Optimising the NumPy implementation

### 7.1 Measure first

Report **µs per simulation step** and the **real-time factor** (simulated time ÷ wall time; with $\Delta t = 1$ ms, anything under 1000 µs/step is faster than real time). Use `time.perf_counter()` around a few thousand steps, `cProfile` to find hot functions, and `line_profiler` (`kernprof -l`) to find hot lines.

There are two regimes, and they need different fixes:

- **Small networks (≲ 500 neurons): Python/NumPy call overhead dominates.** Each NumPy call costs roughly 1–5 µs regardless of array size. The reference robot network runs at about 85 µs/step, and profiling shows about 50 Python-level function calls per step plus dozens of ufunc operations. The arithmetic itself is negligible.
- **Large networks: memory bandwidth dominates.** Now it is about touching as few bytes as possible.

### 7.2 Benchmark: one plastic projection, three ways

One $N \times N$ projection with trace STDP, pre and post neurons firing at 10 Hz, $\Delta t = 1$ ms, float32, single CPU core. All three versions were checked to produce identical weights.

| N | Dense (`outer` every step) | Event-driven NumPy | Numba (fused loops) |
| --- | --- | --- | --- |
| 100 | 32 µs | 10 µs | 0.8 µs |
| 1,000 | 3,232 µs | 70 µs | 24 µs |
| 4,000 | 72,774 µs | 1,903 µs | 476 µs |

Two takeaways:

- Event-driven indexing beats dense by **3× at N=100 and 38–46× at N ≥ 1000**. With a spike probability $p$ per step, dense does $N^2$ work where event-driven does about $2pN^2$.
- Numba is another **3–4× at large N and \~13× at small N**, where it removes the per-call overhead entirely.

### 7.3 The checklist, roughly in order of payoff

**1. Never go dense on spikes.** Replace `s @ W` with `W[idx].sum(0)` and `np.outer(...)` updates with row/column updates on spike indices. This is the single largest win at any realistic size.

**2. Precompute every decay factor.** `d = np.float32(np.exp(-dt/tau))` once at construction; `x *= d` in the loop. Never call `np.exp` per step.

**3. Use float32.** It halves memory traffic, which matters once weights exceed cache. Watch for silent upcasts: `f32_array * 0.01` stays float32 in NumPy 2, but mixing with float64 arrays upcasts. Precision is not a problem for STDP at these magnitudes.

**4. Work in place, avoid temporaries.** `x *= d` instead of `x = x * d`; `np.clip(W, 0, wmax, out=W)`; `np.multiply(a, b, out=tmp)` with a preallocated `tmp`. Every temporary is an allocation plus an extra memory pass.

**5. Merge state to cut call counts (small-network regime).** The reference code decays nine separate arrays with nine calls. Instead:

- Stack all traces with the same τ into one array, or keep one 2-D `traces` array plus a matching `decays` array, and decay everything with one `traces *= decays`.
- If $\tau_+ = \tau_-$ (as in the robot), the pre trace and post trace of a neuron are the *same variable*: store one.
- Put E and I into a single state vector (`v`, `g_e`, `g_i`, `ref` of length $N_E + N_I$) with per-neuron parameter vectors (`tau_m`, `t_ref`) and a single block weight matrix. One LIF call instead of two, one propagation instead of four. You still apply plasticity to the relevant sub-blocks via slicing, which is a view, not a copy.

**6. Mind the memory layout.** With `W[pre, post]` in C order, pre-spike work (propagation, depression) touches contiguous rows while post-spike potentiation touches strided columns. Options: accept it (columns are touched only for post spikes); in Numba, sweep rows in the outer loop and the spiking columns in the inner loop (as in `bench.py`), which reads each row once; or, for very large networks, keep the weights in CSR for pre-driven work plus a CSC index map into the same data array for post-driven work (this is how Brian2 and GeNN do it). Never keep two independent copies of W; they drift apart.

**7. Sparse connectivity.** At $p \le 0.1$, a dense matrix wastes 90% of the traffic and the masks cost another pass. Use a fixed fan-in representation (`pre_idx[post, k]`, `w[post, k]`), or `scipy.sparse` CSR for propagation. For plasticity, a Numba kernel over the sparse structure is much simpler than fighting `scipy.sparse` fancy indexing, which is slow for in-place updates.

**8. Generate randomness in blocks.** `rng.random((T_ctrl, N)) < rates * dt` once per control step instead of once per SNN step: one RNG call instead of 20.

**9. Batch over robots.** If you train many agents (hyperparameter sweeps, evolution, several environments), step them all together with a leading batch axis. Measured on the robot-sized projection (40 × 80, per-robot plastic weights, dense batched ops):

| Robots in batch | Total µs/step | µs/step per robot |
| --- | --- | --- |
| 1 | 17.3 | 17.3 |
| 8 | 68.2 | 8.5 |
| 64 | 627 | 9.8 |
| 256 | 3,059 | 12.0 |

Batching amortises call overhead about 2× here; the sweet spot is where the batch still fits in cache (per-robot cost rises again at 256). With *shared* weights (evaluating one controller in many environments), batched propagation becomes a true matrix–matrix product, and the gain is much larger. Note that in the batched case dense ops are fine: event-driven indexing does not batch well because each robot spikes differently.

```python
def batched_step(W, xpre, xpost, g, s_pre, s_post, tmp):   # W: (B, Npre, Npost)
    xpre *= dp; xpost *= dp
    g += np.einsum('bi,bij->bj', s_pre, W)
    np.multiply(s_pre[:, :, None], xpost[:, None, :], out=tmp); tmp *= W;        W -= Am * tmp
    xpre += s_pre; xpost += s_post
    np.multiply(xpre[:, :, None], s_post[:, None, :], out=tmp); tmp *= (wmax - W); W += Ap * tmp
```

**10. Numba for the inner step.** Once the NumPy version is correct, keep it as the reference and write an `@njit(cache=True, fastmath=True)` kernel that fuses decay, integration, propagation and plasticity into plain loops (full example in `bench.py`). Rules: pass arrays, not Python objects or dataclasses; compare against the NumPy reference with `np.allclose`; parallelise with `prange` over the *batch* axis or over *post* neurons, never over pre spikes that scatter into the same output (race condition).

**11. Lazy (event-driven) traces.** Instead of decaying every trace every step, store each trace's value and last update time, and decay on read: $x(t) = x(t_{last}),e^{-(t - t_{last})/\tau}$. That makes cost proportional to spikes only. It wins for very sparse activity or very large N; at robot scale the per-step `x *= d` is cheaper than the bookkeeping.

**12. Approximations, knowingly.** Updating plasticity every $k$ steps from accumulated spike counts, or applying reward every control step rather than every SNN step (as we do), are valid speed-ups but change the rule. Re-check the learning result after each one.

**13. When to leave NumPy.** JAX with `jit` + `vmap` (batch) + `lax.scan` (time loop) removes Python from the loop and runs on GPU. CuPy is a near drop-in for the NumPy code but only pays off at tens of thousands of neurons or large batches. For big models, dedicated simulators (Brian2 with code generation, GeNN, NEST) already implement everything in this module.

### 7.4 Suggested optimisation path for this project

1. Keep `snn_robot.py` as the correctness reference; save a weight snapshot after a fixed-seed run.
2. Apply items 2–5 and 8 (merge populations and traces, in-place ops, block RNG). Verify the snapshot matches.
3. Port `SNN.step` to one Numba kernel (item 10). Expect roughly an order of magnitude at this size.
4. If you need many agents, add a batch axis and `prange` over it.

---

## Module 8 — Exercises

1. **Window shape.** Plot $\Delta w$ vs $\Delta t$ produced by the trace implementation by simulating pairs of spikes; confirm it matches the analytic window. Then switch to nearest-spike traces and repeat at 50 Hz pairings.
2. **Drift check.** Drive one synapse with independent Poisson pre/post trains and compare the measured drift to $\rho_{pre}\rho_{post}(A_+\tau_+ - A_-\tau_-)$.
3. **iSTDP set point.** Run the network with learning off and different $\rho_0$ values; plot the steady-state E rate against $\rho_0$.
4. **Obstacle avoidance.** Add two proximity sensors and a negative reward on collision. Which wiring should be learned (Braitenberg 3a/3b)?
5. **Enable E→E STDP** (`plastic_EE=True`). Does recurrence help or hurt the controller? Add normalisation to E→E if weights run away.
6. **Triplet rule.** Replace pair STDP with the triplet rule and compare learning speed.
7. **Speed.** Implement the merged-population version and the Numba kernel; report µs/step before and after, and verify identical weights at a fixed seed.

---

## References

- Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons. *J. Neurosci.*
- Song, S., Miller, K. & Abbott, L. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. *Nat. Neurosci.*
- Morrison, A., Diesmann, M. & Gerstner, W. (2008). Phenomenological models of synaptic plasticity based on spike timing. *Biol. Cybern.* (trace formulations, all-to-all vs nearest)
- Pfister, J.-P. & Gerstner, W. (2006). Triplets of spikes in a model of STDP. *J. Neurosci.*
- Vogels, T. P., Sprekeler, H., Zenke, F., Clopath, C. & Gerstner, W. (2011). Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. *Science.*
- Izhikevich, E. M. (2007). Solving the distal reward problem through linkage of STDP and dopamine signaling. *Cereb. Cortex.*
- Frémaux, N. & Gerstner, W. (2016). Neuromodulated STDP and theory of three-factor learning rules. *Front. Neural Circuits.*
- Diehl, P. U. & Cook, M. (2015). Unsupervised learning of digit recognition using STDP. *Front. Comput. Neurosci.*
- Braitenberg, V. (1984). *Vehicles: Experiments in Synthetic Psychology.* MIT Press.
- Bing, Z. et al. (2018). A survey of robotics control based on learning-inspired spiking neural networks. *Front. Neurorobotics.*

---

## Appendix A — `snn_robot.py`

Run `python snn_robot.py`. It trains for 6,000 control steps with progress printed every 500, then reports the frozen test score.

```python
"""E/I spiking network with trace-based STDP, iSTDP and reward-modulated STDP
driving a simulated light-seeking differential-drive robot."""
import numpy as np
from dataclasses import dataclass

f32 = np.float32

@dataclass
class Params:
    dt: float = 1.0            # ms
    # LIF
    v_rest: float = -65.0; v_reset: float = -65.0; v_th: float = -52.0
    tau_mE: float = 20.0; tau_mI: float = 10.0
    t_refE: float = 2.0;  t_refI: float = 1.0
    tau_ge: float = 5.0;  tau_gi: float = 10.0
    I_bias_E: float = 10.0     # tonic drive (mV) so E never goes fully silent
    # excitatory STDP (pair-based, soft bounds)
    tau_plus: float = 20.0; tau_minus: float = 20.0
    A_plus: float = 0.01;  A_minus: float = 0.008   # A-/A+ < 1: slightly Hebbian eligibility
    wmax_in: float = 8.0;  wmax_EE: float = 2.0
    # inhibitory STDP (Vogels et al. 2011)
    tau_istdp: float = 20.0
    eta_i: float = 0.005
    rho0: float = 0.020        # target E rate, spikes/ms (= 20 Hz)
    wmax_IE: float = 50.0
    # three-factor (reward-modulated) learning
    tau_elig: float = 40.0     # ~2 control steps: tight credit assignment
    lr_reward: float = 30.0
    # connectivity
    p_EE: float = 0.1; p_EI: float = 0.3; p_IE: float = 0.4; p_II: float = 0.2
    w_EI: float = 4.0; w_IE0: float = 2.0; w_II: float = 1.0


class SNN:
    """E/I LIF network. Weight matrices are stored as W[pre, post]."""
    def __init__(self, p, n_in, n_e, n_i, seed=0, reward_mode=True, plastic_EE=True):
        self.p, self.rng = p, np.random.default_rng(seed)
        rng, dt = self.rng, p.dt
        self.reward_mode, self.plastic_EE = reward_mode, plastic_EE
        self.n_in, self.n_e, self.n_i = n_in, n_e, n_i
        # --- state
        self.vE = np.full(n_e, p.v_rest, f32); self.vI = np.full(n_i, p.v_rest, f32)
        self.geE = np.zeros(n_e, f32); self.giE = np.zeros(n_e, f32)
        self.geI = np.zeros(n_i, f32); self.giI = np.zeros(n_i, f32)
        self.refE = np.zeros(n_e, f32); self.refI = np.zeros(n_i, f32)
        # --- traces
        self.x_in   = np.zeros(n_in, f32)  # pre trace of inputs      (tau_plus)
        self.xE_pre = np.zeros(n_e, f32)   # pre trace of E for E->E  (tau_plus)
        self.xE_post= np.zeros(n_e, f32)   # post trace of E          (tau_minus)
        self.xI     = np.zeros(n_i, f32)   # pre trace of I  (iSTDP)  (tau_istdp)
        self.xE_i   = np.zeros(n_e, f32)   # post trace of E (iSTDP)  (tau_istdp)
        # --- weights
        self.W_inE = (rng.uniform(0.2, 0.5, (n_in, n_e)) * p.wmax_in).astype(f32)
        self.M_EE = rng.random((n_e, n_e)) < p.p_EE; np.fill_diagonal(self.M_EE, False)
        self.W_EE = (self.M_EE * rng.uniform(0, 0.5, (n_e, n_e)) * p.wmax_EE).astype(f32)
        self.W_EI = ((rng.random((n_e, n_i)) < p.p_EI) * p.w_EI).astype(f32)
        self.M_IE = rng.random((n_i, n_e)) < p.p_IE
        self.W_IE = (self.M_IE * p.w_IE0).astype(f32)
        self.W_II = ((rng.random((n_i, n_i)) < p.p_II) * p.w_II).astype(f32)
        np.fill_diagonal(self.W_II, 0)
        self.elig = np.zeros_like(self.W_inE)
        self.in_sum = self.W_inE.sum(0, keepdims=True).mean()   # target column sum
        # --- precomputed decay factors (never call exp() inside the loop)
        e = lambda tau: f32(np.exp(-dt / tau))
        self.dmE, self.dmI = e(p.tau_mE), e(p.tau_mI)
        self.dge, self.dgi = e(p.tau_ge), e(p.tau_gi)
        self.dplus, self.dminus, self.distdp = e(p.tau_plus), e(p.tau_minus), e(p.tau_istdp)
        self.delig = e(p.tau_elig)
        self.alpha = f32(2 * p.rho0 * p.tau_istdp)   # iSTDP depression offset

    # ------------------------------------------------------------------
    def _lif(self, v, ge, gi, ref, decay, t_ref, bias=0.0):
        p = self.p
        v_inf = p.v_rest + bias + ge - gi                  # current-based drive, in mV
        v[:] = v_inf + (v - v_inf) * decay          # exact integration over dt
        ref -= p.dt
        v[ref > 0] = p.v_reset                      # refractory clamp
        spk = v >= p.v_th
        v[spk] = p.v_reset; ref[spk] = t_ref
        return np.flatnonzero(spk)

    def step(self, in_spk_idx):
        p = self.p
        # 1) decay everything
        for x, d in ((self.x_in, self.dplus), (self.xE_pre, self.dplus),
                     (self.xE_post, self.dminus), (self.xI, self.distdp),
                     (self.xE_i, self.distdp), (self.geE, self.dge),
                     (self.geI, self.dge), (self.giE, self.dgi), (self.giI, self.dgi)):
            x *= d
        if self.reward_mode: self.elig *= self.delig

        # 2) integrate neurons
        e_idx = self._lif(self.vE, self.geE, self.giE, self.refE, self.dmE, p.t_refE, p.I_bias_E)
        i_idx = self._lif(self.vI, self.geI, self.giI, self.refI, self.dmI, p.t_refI)

        # 3) propagate spikes (event-driven: sum rows of the spiking pre neurons)
        self.geE += self.W_inE[in_spk_idx].sum(0) + self.W_EE[e_idx].sum(0)
        self.geI += self.W_EI[e_idx].sum(0)
        self.giE += self.W_IE[i_idx].sum(0)
        self.giI += self.W_II[i_idx].sum(0)

        # 4) plasticity
        self._exc_stdp(in_spk_idx, e_idx)
        self._inh_stdp(i_idx, e_idx)
        return e_idx, i_idx

    # ---- excitatory STDP with traces --------------------------------
    def _exc_stdp(self, in_idx, e_idx):
        p = self.p
        # (a) pre spikes -> depression, uses post trace BEFORE this step's post spikes
        W_in = self.W_inE
        dep_in = -p.A_minus * self.xE_post[None, :] * W_in[in_idx]          # soft bound: * w
        if self.plastic_EE and len(e_idx):
            dep_EE = -p.A_minus * self.xE_post[None, :] * self.W_EE[e_idx]
            self.W_EE[e_idx] += dep_EE * self.M_EE[e_idx]
        # (b) update traces (all-to-all: += 1 ; nearest-spike: = 1)
        self.x_in[in_idx] += 1; self.xE_pre[e_idx] += 1; self.xE_post[e_idx] += 1
        # (c) post spikes -> potentiation, uses pre trace
        pot_in = p.A_plus * self.x_in[:, None] * (p.wmax_in - W_in[:, e_idx])  # soft bound: * (wmax-w)
        if self.plastic_EE and len(e_idx):
            pot_EE = p.A_plus * self.xE_pre[:, None] * (p.wmax_EE - self.W_EE[:, e_idx])
            self.W_EE[:, e_idx] += pot_EE * self.M_EE[:, e_idx]
        target = self.elig if self.reward_mode else W_in   # three-factor: write to eligibility
        target[in_idx] += dep_in
        target[:, e_idx] += pot_in

    # ---- inhibitory STDP (Vogels-Sprekeler) -------------------------
    def _inh_stdp(self, i_idx, e_idx):
        p = self.p
        self.xI[i_idx] += 1; self.xE_i[e_idx] += 1
        W, M = self.W_IE, self.M_IE
        if len(i_idx):   # pre (I) spike: w += eta * (x_post - alpha)
            W[i_idx] += p.eta_i * (self.xE_i[None, :] - self.alpha) * M[i_idx]
        if len(e_idx):   # post (E) spike: w += eta * x_pre
            W[:, e_idx] += p.eta_i * self.xI[:, None] * M[:, e_idx]
        np.clip(W, 0, p.wmax_IE, out=W)

    # ---- third factor ------------------------------------------------
    def apply_reward(self, delta_r):
        W = self.W_inE
        W += self.p.lr_reward * delta_r * self.elig
        np.clip(W, 0, self.p.wmax_in, out=W)
        # synaptic normalisation: fixed total input per post neuron (kills common-mode drift)
        W *= self.in_sum / (W.sum(0, keepdims=True) + 1e-9)


class Vehicle:
    """Differential-drive robot seeking a light source."""
    def __init__(self, rng, arena=10.0, sensor_angle=np.pi / 4, L=0.5):
        self.rng, self.arena, self.sa, self.L = rng, arena, sensor_angle, L
        self.reset()

    def reset(self):
        r = self.rng
        self.pos = r.uniform(-self.arena, self.arena, 2)
        self.th = r.uniform(-np.pi, np.pi)
        self.target = r.uniform(-self.arena, self.arena, 2)

    def bearing(self):
        d = self.target - self.pos
        return (np.arctan2(d[1], d[0]) - self.th + np.pi) % (2 * np.pi) - np.pi

    def sensors(self):
        b = self.bearing()          # >0 : target on the left
        return 0.5 * (1 + np.cos([b - self.sa, b + self.sa]))   # in [0, 1], never fully blind

    def move(self, vl, vr, dt_s):
        v, w = (vl + vr) / 2, (vr - vl) / self.L
        self.th += w * dt_s
        self.pos += v * dt_s * np.array([np.cos(self.th), np.sin(self.th)])
        if np.linalg.norm(self.target - self.pos) < 1.0:
            self.target = self.rng.uniform(-self.arena, self.arena, 2)


def run(n_ctrl=3000, seed=0, learn=True, net=None, verbose=True, gain=0.05, v0=0.3, **kw):
    p = Params(**kw)
    n_per_sensor, n_e, n_i, n_motor = 20, 80, 20, 40
    rng = np.random.default_rng(seed + 1)
    if net is None:
        net = SNN(p, 2 * n_per_sensor, n_e, n_i, seed=seed, reward_mode=True, plastic_EE=False)
    bot = Vehicle(rng)
    T_ctrl, r_max = 20, 0.08           # 20 ms per control step, 80 Hz max input rate
    r_bar, err_prev, log = 0.0, abs(bot.bearing()), []
    for k in range(n_ctrl):
        s = bot.sensors()
        rates = np.repeat(0.01 + s * r_max, n_per_sensor)      # spikes/ms (10 Hz floor)
        in_spk = rng.random((T_ctrl, rates.size)) < rates * p.dt  # pre-generate a block
        cnt = np.zeros(n_e)
        for t in range(T_ctrl):
            e_idx, _ = net.step(np.flatnonzero(in_spk[t]))
            cnt[e_idx] += 1
        rL = cnt[:n_motor].mean() / T_ctrl * 1000              # Hz
        rR = cnt[n_motor:2 * n_motor].mean() / T_ctrl * 1000
        turn = gain * (rR - rL)                                 # push-pull decoding
        vl, vr = v0 - turn, v0 + turn
        bot.move(vl, vr, T_ctrl / 1000)
        err = abs(bot.bearing())
        r = err_prev - err; err_prev = err        # reward = reduction of |bearing error|
        align = np.cos(bot.bearing())
        if learn:
            net.apply_reward(r - r_bar)
        r_bar += 0.05 * (r - r_bar)
        log.append(align)
        if verbose and (k + 1) % 500 == 0:
            print(f"ctrl {k+1:5d}  mean cos(bearing) last 500 = {np.mean(log[-500:]):+.3f}  "
                  f"E rate ~ {cnt.mean()/T_ctrl*1000:5.1f} Hz  W_IE mean {net.W_IE[net.M_IE].mean():.2f}")
    return net, np.array(log)

if __name__ == "__main__":
    p = Params()
    untrained = SNN(p, 40, 80, 20, seed=0, plastic_EE=False)
    _, before = run(1500, seed=100, learn=False, net=untrained, verbose=False)
    net, _ = run(6000, seed=0)
    _, after = run(1500, seed=100, learn=False, net=net, verbose=False)
    print(f"frozen test: cos(bearing) {before.mean():+.2f} (untrained) -> {after.mean():+.2f} (trained)")
```

## Appendix B — `bench.py`

Requires `pip install numba`. Prints the Module 7 benchmark table.

```python
"""Benchmark one plastic projection (pre -> post) with trace STDP, 4 ways."""
import numpy as np, time
from numba import njit
f32 = np.float32
Ap, Am, wmax = f32(0.01), f32(0.008), f32(1.0)
dp = f32(np.exp(-1/20))

def make(N, seed=0):
    rng = np.random.default_rng(seed)
    return dict(W=rng.uniform(0, .5, (N, N)).astype(f32),
                xpre=np.zeros(N, f32), xpost=np.zeros(N, f32), g=np.zeros(N, f32))

# --- v0: dense, clock-driven, allocates every step
def step_dense(S, s_pre, s_post):
    W = S['W']
    S['xpre'] = S['xpre'] * dp; S['xpost'] = S['xpost'] * dp
    S['g'] = S['g'] + s_pre @ W
    W += -Am * np.outer(s_pre, S['xpost']) * W
    S['xpre'] = S['xpre'] + s_pre; S['xpost'] = S['xpost'] + s_post
    W += Ap * np.outer(S['xpre'], s_post) * (wmax - W)

# --- v1: event-driven indexing, in-place
def step_event(S, pre_idx, post_idx):
    W, xpre, xpost, g = S['W'], S['xpre'], S['xpost'], S['g']
    xpre *= dp; xpost *= dp
    if pre_idx.size:
        rows = W[pre_idx]
        g += rows.sum(0)
        rows -= Am * xpost * rows
        W[pre_idx] = rows
    xpre[pre_idx] += 1; xpost[post_idx] += 1
    if post_idx.size:
        cols = W[:, post_idx]
        cols += Ap * xpre[:, None] * (wmax - cols)
        W[:, post_idx] = cols

# --- v2: numba, fused loops, no temporaries
@njit(cache=True, fastmath=True)
def step_numba(W, xpre, xpost, g, pre_idx, post_idx):
    N_pre, N_post = W.shape
    for i in range(N_pre): xpre[i] *= dp
    for j in range(N_post): xpost[j] *= dp
    for k in range(pre_idx.size):                  # pre spikes: propagate + depress row
        i = pre_idx[k]
        for j in range(N_post):
            w = W[i, j]
            g[j] += w
            W[i, j] = w - Am * xpost[j] * w
    for k in range(pre_idx.size): xpre[pre_idx[k]] += 1
    for k in range(post_idx.size): xpost[post_idx[k]] += 1
    for i in range(N_pre):                         # post spikes: potentiate columns
        a = Ap * xpre[i]                           # (row-major sweep: cache friendly)
        if a == 0: continue
        for k in range(post_idx.size):
            j = post_idx[k]
            W[i, j] += a * (wmax - W[i, j])

def bench(N, rate_hz=10, T=2000, reps=1):
    rng = np.random.default_rng(1)
    p = rate_hz / 1000
    pre = rng.random((T, N)) < p; post = rng.random((T, N)) < p
    pre_idx = [np.flatnonzero(r) for r in pre]; post_idx = [np.flatnonzero(r) for r in post]
    pre_f = pre.astype(f32); post_f = post.astype(f32)
    out = {}
    S = make(N); t = time.perf_counter()
    for k in range(T): step_dense(S, pre_f[k], post_f[k])
    out['dense'] = (time.perf_counter() - t) / T * 1e6
    Wd = S['W'].copy()
    S = make(N); t = time.perf_counter()
    for k in range(T): step_event(S, pre_idx[k], post_idx[k])
    out['event'] = (time.perf_counter() - t) / T * 1e6
    assert np.allclose(S['W'], Wd, atol=1e-5), "event != dense"
    S = make(N); step_numba(S['W'], S['xpre'], S['xpost'], S['g'], pre_idx[0], post_idx[0])  # compile
    S = make(N); t = time.perf_counter()
    for k in range(T): step_numba(S['W'], S['xpre'], S['xpost'], S['g'], pre_idx[k], post_idx[k])
    out['numba'] = (time.perf_counter() - t) / T * 1e6
    assert np.allclose(S['W'], Wd, atol=1e-4), "numba != dense"
    return out

if __name__ == "__main__":
    print(f"{'N':>6} {'dense':>10} {'event':>10} {'numba':>10}   (us / step, 10 Hz)")
    for N in [100, 1000, 4000]:
        o = bench(N, T=2000 if N < 4000 else 300)
        print(f"{N:6d} {o['dense']:10.1f} {o['event']:10.1f} {o['numba']:10.1f}")
```
