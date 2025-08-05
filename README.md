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

# Create a virtual environment (Python 3.10)
conda create -n Ctx2TrajGen python=3.10
conda activate Ctx2TrajGen

# Install dependencies
pip install -r requirements.txt
```
