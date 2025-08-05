# Ctx2TrajGen: Traffic Context-Aware Microscale Vehicle Trajectories using Generative Adversarial Imitation Learning

[Joobin Jin](https://github.com/joobinJin), [Seokjun Hong](https://github.com/seokjun-h), [Gyeongseon Baek](modifiying), [Yeeun Kim](modifiying), [Byeongjoon Noh](https://scholar.google.com/citations?hl=ko&user=0mPWzzIAAAAJ)

[[`Paper`](https://arxiv.org/abs/2507.17418)] [[`Dataset`](https://huggingface.co/datasets/Hj-Lee/The-DRIFT)] 

## 🚀 Overview

![Architecture](images/architecture.png)

**Ctx2TrajGen**  is a Generative Adversarial Imitation Learning (GAIL) framework for generating microscale vehicle trajectories in real-world traffic scenes. Our model learns realistic vehicle movement patterns by modeling interactions, road structure, and dynamics, without requiring explicit reward design.


## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/joobinJin/Ctx2TrajGen.git
cd Ctx2TrajGen
```

```bash
# Create a virtual environment (Python 3.10)
conda create -n Ctx2TrajGen python=3.10
conda activate Ctx2TrajGen
```

```bash
# Install dependencies
pip install -r requirements.txt
```
## 🧪 Running Tests

You can verify each core module (Environment, Discriminator, Policy & Value Network) with the following unit test scripts:

### 1. Environment Test

Runs a simple environment rollout to ensure the `MicroTrajectoryEnv` is properly initialized and functional.

```bash
python test/test_env.py
```

### 2. Discriminator Test

Tests whether the `DiscriminatorRNN` can process sample state-action pairs and output realistic discrimination scores.

```bash
python test/test_discriminator.py
```

### 3. Policy & Value Network Test

Runs inference using `PolicyNetRNN` and `ValueNetRNN` to verify model outputs such as action probabilities and state values for a batch of input trajectories.

```bash
python test/test_policy_net.py
```


