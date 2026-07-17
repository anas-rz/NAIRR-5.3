# Lecture 5.3: Edge Computing in AI

**NAIRR Workshop Series 2026**

This session explores edge computing in AI for real-time decision-making, imitation learning, intelligent offloading, and agentic AI for edge medical applications.

---

## Slides

> 🔗 [Slides](slides/nairr5.3.pdf)

---

## Notebooks

| #  | Notebook | Topic                                              |
| --- | -------- | -------------------------------------------------- |
| 00 | NB01     | Agentic AI for Edge-Based Medical Decision Support |
| 01 | NB02     | End-to-End Imitation Learning on Edge Devices    |
| 02 | NB03     | Real-Time Control and Decision-making with Edge       |
| 03 | NB04     | Intelligent Offloading Strategies for Edge AI      |

---

## 📓 Notebook Details

### NB01 — Agentic AI for Edge-Based Medical Decision Support

**Focus:** A compact, on-device **Retrieval-Augmented Generation (RAG)** agent for answering sensitive medical questions without sending data to the cloud.

- Uses **Gemma-2-2B**, an instruction-tuned, 4-bit quantized open-source LLM sized for edge deployment
- Uploads user medical documents, chunks and embeds them, retrieves relevant segments per query, and grounds the LLM's answer in that context
- Illustrates the four core components of an LLM agent: agent code (LLM), memory, tools, and planning
- Emphasizes local, private processing of sensitive clinical records with no external data transmission

### NB02 — End-to-End Imitation Learning on Edge Devices

**Focus:** Training a lightweight neural policy that maps raw sensor perception directly to robot motor commands, replacing brittle hand-written control rules.

- Uses **behavioral cloning** to mimic an expert agent's navigation behavior
- Abstracts raw LiDAR point clouds into compact spatial bins (Front / Left / Right) to cut noise and compute cost
- Trains and compares a baseline **MLP policy** (with dropout) against a **residual policy network** (skip connections + batch normalization)
- Splits training data by episode, not by random step, to prevent the model from memorizing specific room layouts

### NB03 — Real-Time Control and Decision-Making with Edge

**Focus:** Building an edge-style forecasting and decision pipeline for volatile, real-time signals such as household energy usage.

- Compares **Ridge Regression** (linear baseline), **1D-CNN** (lightweight edge model), and **GRU** (sequential model) for time-series forecasting
- Applies robust scaling and windowed preprocessing to handle spikes and outliers in sensor telemetry
- Implements an autonomous **Edge Controller** that simulates real-time load-shedding decisions from the model's predictions
- Discusses deployment to low-power hardware (e.g., Raspberry Pi) via TorchScript/ONNX conversion

### NB04 — Intelligent Offloading Strategies for Edge AI

**Focus:** Deciding, in real time, whether an inference task should be processed locally or offloaded to a remote/cloud worker.

- Trains a **neural policy network** (PyTorch, BatchNorm + Dropout) on real-world telemetry from the **Alibaba Cluster Trace (2026)** — GPU utilization, latency, and bandwidth as state features
- Frames offloading as binary classification and handles class imbalance with **median-based cost scoring**
- Benchmarks the neural policy against **Logistic Regression**, **Random Forest**, and a simple MLP

---

## Local Setup

```
git clone https://github.com/NifulIslam/NAIRR-5.3.git
cd NAIRR-5.3
python -m venv NAIRR-edge
```

**Activate the environment:**

- **Windows:** `NAIRR-edge\Scripts\activate`
- **macOS/Linux:** `source NAIRR-edge/bin/activate`

```
pip install -r requirements.txt
```

Open in VS Code, select the `NAIRR-edge` kernel, and run.

---

## 📊 Key Insights

- **Latency-Critical Control:** Hardware latency in robotics and control loops can destabilize the system; cloud round-trips are too slow for sub-second decisions.
- **Reality vs. Simulation Gap:** Real-world sensor noise creates gaps between simulated training data and deployed behavior.
- **Bursty Workloads:** GenAI traffic bursts can cause severe GPU congestion, motivating adaptive local/cloud hybrid systems.
- **Privacy-Sensitive Data:** Sensitive data (e.g., clinical records) benefits from local, on-device processing rather than cloud transmission.

### Key Performance Lessons

- Feature abstraction (e.g., LiDAR binning) is non-negotiable for real-time edge inference.
- Data-driven neural policies consistently outperform static, rule-based schedulers.
- Robust preprocessing is essential when dealing with noisy physical sensors.
- Prefer smaller, quantized LLMs when processing sensitive data locally.

## Funding

NSF NAIRR-Nat AI Research Resource (NAIRR) Awards OAC-2528533 and OAC-2528534
