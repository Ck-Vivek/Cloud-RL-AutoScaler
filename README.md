# Cloud-RL: AI-Driven Auto-scaler

## Project Overview
This project is an intelligent cloud auto-scaling simulator that scales servers up or down based on real-time traffic requests. It uses advanced Large Language Models (LLMs) to predict and analyze the perfect scaling patterns while balancing running costs and strict SLA (latency) requirements.

## Architecture
The system consists of a **Hybrid Auto-scaler**:
1. **AI-Optimized Control**: Leverages the OpenRouter API to evaluate the state of the cloud (CPU, Latency, Traffic, Cost) and outputs JSON-structured scaling decisions.
2. **Heuristic Fallback (Autonomous Mode)**: A hard-coded rule-based safety net.

### Why sometimes '0 Tokens'?
Since we use free-tier APIs, the system might encounter `429 Too Many Requests` or API downtimes. In such cases, our system **does not crash**; instead, it seamlessly switches to the 'Autonomous Mode' safety-net to ensure 100% uptime and 0 SLA downtime. If you see '0 Tokens' used, this demonstrates the system's absolute resilience and fault-tolerance against external network failures!

## Task Levels
The simulation tests robustness against three traffic conditions:
* **Easy**: Stable, predictable request loads (40-50 req/s). Ideal for checking cost minimization.
* **Medium**: Sine-wave patterns that test predictive scaling up and down dynamically.
* **Hard**: Chaotic high-burst spikes (random traffic bounds, with an extreme burst up to 350 req/s at Step 25) to test the agent's SLA breach prevention limits!
