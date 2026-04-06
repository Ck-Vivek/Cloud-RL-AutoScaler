import os
import sys
import json
import time
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import numpy as np
from tabulate import tabulate

from models import CloudObservation, CloudAction, CloudReward, CloudEpisodeSummary
from env import CloudScalingEnv


# Load environment variables at the very top
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", "")
API_BASE_URL = 'https://openrouter.ai/api/v1'
MODEL_NAME = 'google/gemini-2.0-flash-exp:free'


class CloudAgent:
    """AI Agent for cloud scaling decisions using OpenRouter."""

    def __init__(self, model: str = MODEL_NAME, api_key: str = None):
        """
        Initialize the CloudAgent.
        
        Args:
            model: Model name (default from .env)
            api_key: API key (default from .env HF_TOKEN)
        """
        self.model = model
        self.api_key = api_key or HF_TOKEN
        if not self.api_key:
            print('⚠️ API Key is missing! Check your .env file.')
        self.client = OpenAI(api_key=self.api_key, base_url=API_BASE_URL)
        self.tokens_used = 0
        print(f'Using Model: {self.model}')
    
    def _build_prompt(self, obs: CloudObservation) -> str:
        """Build a compact prompt for the agent."""
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
        """
        Clean JSON response by removing markdown code blocks.
        
        Args:
            content: Raw response content
            
        Returns:
            Cleaned JSON string
        """
        # Strip leading/trailing whitespace
        content = content.strip()
        
        # Remove markdown code block markers (```json ... ``` or ``` ... ```)
        if content.startswith("```"):
            # Remove opening ```json, ```json, or ```
            content = content.lstrip("`")
            # Remove language identifier if present (e.g., "json")
            lines = content.split("\n", 1)
            if len(lines) > 1:
                content = lines[1]
        
        if content.endswith("```"):
            content = content.rstrip("`")
        
        return content.strip()
    
    def _fallback_action(self, obs: CloudObservation) -> CloudAction:
        """Heuristic fallback for API failures."""
        if obs.lat > 100.0 or obs.cpu_util > 80.0:
            return CloudAction(action_type=1, delta=min(3, 50 - obs.n_servers))
        elif obs.cpu_util < 30.0 and obs.lat < 60.0 and obs.n_servers > 1:
            return CloudAction(action_type=2, delta=1)
        else:
            return CloudAction(action_type=0, delta=0)
    
    def get_action(self, obs: CloudObservation) -> CloudAction:
        """
        Get scaling action from OpenRouter AI agent.
        
        Args:
            obs: Current CloudObservation
            
        Returns:
            CloudAction
        """
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
            
            # Track tokens
            if hasattr(resp, 'usage') and resp.usage is not None:
                step_tokens = resp.usage.total_tokens
                self.tokens_used += step_tokens
                import streamlit as st
                st.write(f'Tokens used this step: {step_tokens}')
            
            # Get response content
            raw_content = resp.choices[0].message.content
            if raw_content is None:
                raise ValueError("API returned None content")
            content = raw_content.strip()
            
            # Clean markdown markers
            content = self._clean_json_response(content)
            
            # Extract only JSON if extra text surrounds it
            if '{' in content and '}' in content:
                content = content[content.find('{'):content.rfind('}')+1]
            
            # DEBUG: Show raw AI response
            print(f'DEBUG: AI replied with -> {content}')
            
            # Parse JSON with specific error handling
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"  [Fallback] JSON parse error: {e}. Using heuristic.", file=sys.stderr)
                return self._fallback_action(obs)
            
            # Validate and build action
            try:
                action = CloudAction(
                    action_type=int(data.get("action_type", 0)),
                    delta=int(data.get("delta", 0)),
                    target_servers=data.get("target_servers"),
                )
                return action
            except (KeyError, ValueError, TypeError) as e:
                print(f"  [Fallback] Action validation error: {e}. Using heuristic.", file=sys.stderr)
                return self._fallback_action(obs)
        
        except Exception as e:
            if '429' in str(e):
                print('⚠️ Rate limit hit! Waiting for 5 seconds...')
                time.sleep(5)
            else:
                print(f'API Error: {e}')
            return self._fallback_action(obs)
    
    def get_tokens_used(self) -> int:
        """Return total tokens used."""
        return self.tokens_used


def main():
    """Main simulation loop."""
    print("=" * 100)
    print("Cloud Scaling Agent - Simulation")
    print("=" * 100)
    
    # Initialize
    env = CloudScalingEnv(
        init_servers=5,
        max_servers=50,
        min_servers=1,
        max_steps=50,
        base_cost=0.05,
        seed=42,
    )
    
    agent = CloudAgent()
    
    # Reset environment
    obs = env.reset()
    
    # Tracking  
    total_cost = 0.0
    total_reward = 0.0
    sla_breaches = 0  # Latency > 200ms
    table_data = []
    
    print()
    print("Starting simulation...")
    print()
    
    # Main loop
    for step in range(env.max_steps):
        time.sleep(1)
        
        # Get action from agent
        action = agent.get_action(obs)
        
        # Step environment
        obs_new, rew, done = env.step(action)
        
        # Track metrics
        total_cost += obs_new.cost_hr
        total_reward += rew.rew
        if obs_new.lat > 200.0:
            sla_breaches += 1
        
        # Action type name
        action_name = ["Hold", "Up", "Down"][action.action_type]
        if action.action_type in (1, 2):
            action_name += f" (+{action.delta})" if action.action_type == 1 else f" (-{action.delta})"
        
        # Add to table
        table_data.append([
            obs_new.step,
            action_name,
            obs_new.n_servers,
            obs_new.req_count,
            f"{obs_new.lat:.1f}ms",
            f"{rew.rew:.2f}",
        ])
        
        obs = obs_new
        
        if done:
            print(f"Episode ended at step {step + 1}.")
            break
    
    # Print table
    print()
    print(tabulate(
        table_data,
        headers=["Step", "Action", "Servers", "Traffic", "Latency", "Reward"],
        tablefmt="grid",
    ))
    print()
    
    # Summary
    summary = CloudEpisodeSummary(
        score=total_reward,
        total_cost=total_cost,
        tokens_used=agent.get_tokens_used(),
        sla_breaches=sla_breaches,
        n_steps=obs.step + 1,
    )
    
    print("=" * 100)
    print("EPISODE SUMMARY")
    print("=" * 100)
    print(f"Total Steps:       {summary.n_steps}")
    print(f"Total Reward:      {summary.score:.2f}")
    print(f"Total Cost:        ${summary.total_cost:.2f}")
    print(f"Tokens Used:       {summary.tokens_used}")
    print(f"SLA Breaches:      {summary.sla_breaches}")
    print("=" * 100)


if __name__ == "__main__":
    main()
