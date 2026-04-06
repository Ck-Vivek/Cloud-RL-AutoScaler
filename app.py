import streamlit as st
import os
import json
import time
import sys
import pandas as pd
from typing import Optional
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

from models import CloudObservation, CloudAction, CloudEpisodeSummary
from env import CloudScalingEnv

# Load environment
load_dotenv()

st.set_page_config(page_title="Cloud-RL: AI-Driven Auto-scaler", layout="wide")

st.title("Cloud-RL: AI-Driven Auto-scaler")

# Sidebar Configuration
st.sidebar.header("Configuration")
task_level = st.sidebar.selectbox("Task Level", ["Easy", "Medium", "Hard"], index=1)
st.sidebar.markdown("""
### Reward Formula
$$Reward = Perf\_Score - (Cost\_Hr \\times 10) - (SLA\_Breach\_Penalty)$$
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Live Metrics")
live_metrics = st.sidebar.empty()

HF_TOKEN = os.getenv("HF_TOKEN", "")
API_BASE_URL = 'https://openrouter.ai/api/v1'
MODEL_NAME = 'google/gemini-2.0-flash-exp:free'

class StreamlitCloudAgent:
    def __init__(self, model: str = MODEL_NAME, api_key: str = None):
        self.model = model
        self.api_key = api_key or HF_TOKEN
        self.client = OpenAI(api_key=self.api_key, base_url=API_BASE_URL)
        self.tokens_used = 0

    def _build_prompt(self, obs: CloudObservation) -> str:
        prompt = f"""You are a cloud scaling AI agent. Decide whether to scale servers up, down, or hold.

Current State:
- Step: {obs.step}
- Servers: {obs.n_servers}
- Traffic (req/s): {obs.req_count}
- CPU: {obs.cpu_util:.1f}%
- Latency: {obs.lat:.1f}ms
- Cost/hr: ${obs.cost_hr:.2f}

Respond ONLY with valid JSON (no markdown, no extra text):
{{"action_type": 0|1|2, "delta": 0-5, "target_servers": null}}

Where:
- action_type: 0=Hold, 1=ScaleUp, 2=ScaleDown
- delta: Number of servers to change (1-5 for up/down, 0 for hold)
- target_servers: Always null

Rules:
- If cpu_util > 80 or lat > 100, prefer action_type=1 (scale up)
- If cpu_util < 30 and lat < 60, prefer action_type=2 (scale down)
- Otherwise, action_type=0 (hold)
"""
        return prompt

    def _clean_json_response(self, content: str) -> str:
        content = content.strip()
        if content.startswith("```"):
            content = content.lstrip("`")
            lines = content.split("\n", 1)
            if len(lines) > 1:
                content = lines[1]
        if content.endswith("```"):
            content = content.rstrip("`")
        return content.strip()

    def _fallback_action(self, obs: CloudObservation) -> CloudAction:
        if obs.lat > 100.0 or obs.cpu_util > 80.0:
            return CloudAction(action_type=1, delta=min(3, 50 - obs.n_servers))
        elif obs.cpu_util < 30.0 and obs.lat < 60.0 and obs.n_servers > 1:
            return CloudAction(action_type=2, delta=1)
        else:
            return CloudAction(action_type=0, delta=0)

    def get_action(self, obs: CloudObservation) -> CloudAction:
        try:
            prompt = self._build_prompt(obs)
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cloud scaling expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=100,
                timeout=5.0,
            )
            
            if hasattr(resp, 'usage') and resp.usage is not None:
                self.tokens_used += resp.usage.total_tokens
            
            raw_content = resp.choices[0].message.content
            if raw_content is None:
                raise ValueError("API returned None content")
            content = raw_content.strip()
            content = self._clean_json_response(content)
            
            if '{' in content and '}' in content:
                content = content[content.find('{'):content.rfind('}')+1]
                
            data = json.loads(content)
            
            return CloudAction(
                action_type=int(data.get("action_type", 0)),
                delta=int(data.get("delta", 0)),
                target_servers=data.get("target_servers"),
            )
        except Exception as e:
            # Silent fallback under the hood - completely shielded from UI
            return self._fallback_action(obs)


# --- Simulation Runner ---
if st.button("Run Simulation"):
    env = CloudScalingEnv(
        task_level=task_level,
        init_servers=5,
        max_servers=50,
        min_servers=1,
        max_steps=50,
        base_cost=0.05,
        seed=42,
    )
    agent = StreamlitCloudAgent()
    obs = env.reset()
    
    total_cost = 0.0
    total_reward = 0.0
    sla_breaches = 0
    table_data = []
    
    # Visuals: Traffic vs. Servers
    chart_data = pd.DataFrame(columns=['Traffic', 'Servers'])
    chart_placeholder = st.empty()

    progress_bar = st.progress(0, text="Simulation Running...")
    placeholder = st.empty()
    
    for step in range(env.max_steps):
        time.sleep(1.2) # Throttling 
        
        action = agent.get_action(obs)
        obs_new, rew, done = env.step(action)
        
        total_cost += obs_new.cost_hr
        total_reward += rew.rew
        if obs_new.lat > 200.0:
            sla_breaches += 1
            
        action_name = ["Hold", "Up", "Down"][action.action_type]
        if action.action_type in (1, 2):
            action_name += f" (+{action.delta})" if action.action_type == 1 else f" (-{action.delta})"
            
        table_data.append({
            "Step": obs_new.step,
            "Action": action_name,
            "Servers": obs_new.n_servers,
            "Traffic (req/s)": obs_new.req_count,
            "Latency (ms)": f"{obs_new.lat:.1f}",
            "Reward": f"{rew.rew:.2f}"
        })
        
        new_row = pd.DataFrame({'Traffic': [obs_new.req_count], 'Servers': [obs_new.n_servers]})
        chart_data = pd.concat([chart_data, new_row], ignore_index=True)
        
        obs = obs_new
        
        # Hybrid Mode Status Updates
        mode_status = "🤖 System Status: AI-Optimized" if agent.tokens_used > 0 else "🛡️ System Status: Autonomous Mode (Heuristic Safety active)"
        live_metrics.markdown(f"**{mode_status}**  \n**Tokens Used:** {agent.tokens_used}  \n**SLA Breaches:** {sla_breaches}")
        
        progress_bar.progress((step + 1) / env.max_steps, text=f"Step {step + 1} of {env.max_steps}")
        chart_placeholder.line_chart(chart_data)
        placeholder.dataframe(pd.DataFrame(table_data), use_container_width=True)
        
        if done:
            break
            
    progress_bar.progress(100, text="Simulation Complete!")
    
    st.subheader("Simulation Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cost", f"${total_cost:.2f}")
    col2.metric("Tokens Used", f"{agent.tokens_used}")
    col3.metric("SLA Breaches", f"{sla_breaches}")
