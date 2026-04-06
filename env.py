import numpy as np
from models import CloudObservation, CloudAction, CloudReward


class CloudScalingEnv:
    """Simple cloud scaling environment simulator."""

    def __init__(
        self,
        task_level: str = "Medium",
        init_servers: int = 5,
        max_servers: int = 50,
        min_servers: int = 1,
        max_steps: int = 50,
        base_cost: float = 0.05,
        seed: int = None,
    ):
        """
        Initialize the environment.
        """
        self.task_level = task_level
        self.init_servers = init_servers
        self.max_servers = max_servers
        self.min_servers = min_servers
        self.max_steps = max_steps
        self.base_cost = base_cost
        
        if seed is not None:
            np.random.seed(seed)
        
        self.step_count = 0
        self.n_servers = init_servers
        self.req_count = 0
        
    def _gen_traffic(self) -> int:
        """Generate request count using task-level logic."""
        if self.task_level == "Easy":
            req = int(40 + np.random.uniform(-5, 5))
        elif self.task_level == "Hard":
            if self.step_count == 25:
                # Guaranteed SLA Breach spike on Step 25
                req = 350
            else:
                base = 40
                spike = 0
                if np.random.random() < 0.15: 
                    spike = np.random.randint(110, 150)
                req = int(max(0, base + spike))
        else: # Medium
            base = 50 * (1 + np.sin(self.step_count / 50))
            spike = np.random.normal(0, 10)
            req = int(max(0, base + spike))
        return req
    
    def _calc_cpu_util(self, req_count: int, n_servers: int) -> float:
        """Calculate CPU utilization: req_count / n_servers."""
        return min(100.0, (req_count / n_servers) * 10.0)
    
    def _calc_lat(self, cpu_util: float) -> float:
        """
        Calculate latency.
        Base latency: 50ms.
        Spike if cpu_util > 80%: latency = 50 + (cpu_util - 80) * 2
        """
        base_lat = 50.0
        if cpu_util > 80.0:
            lat = base_lat + (cpu_util - 80.0) * 2.0
        else:
            lat = base_lat + np.random.normal(0, 5)
        return max(base_lat, lat)
    
    def _calc_cost_hr(self, n_servers: int) -> float:
        """Calculate hourly cost: base_cost * n_servers."""
        return self.base_cost * n_servers
    
    def _calc_perf_score(self, cpu_util: float, lat: float) -> float:
        """
        Calculate performance score.
        Higher is better. Penalize high latency and high CPU.
        """
        cpu_score = max(0, 100 - cpu_util)
        lat_score = max(0, 200 - lat)
        return (cpu_score + lat_score) / 2.0
    
    def _calc_reward(self, obs_prev, obs_curr, cost_hr: float) -> CloudReward:
        """
        Calculate reward.
        Reward = Performance Score - Cost Penalty - SLA Breach Penalty
        """
        perf_score = self._calc_perf_score(obs_curr.cpu_util, obs_curr.lat)
        sla_breach_pen = 100.0 if obs_curr.lat > 200.0 else 0.0
        cost_pen = cost_hr * 10
        act_pen = 0.0
        rew = perf_score - cost_pen - sla_breach_pen
        
        return CloudReward(
            rew=rew,
            cost_pen=cost_pen,
            lat_pen=sla_breach_pen,
            act_pen=act_pen,
        )
    
    def step(self, action: CloudAction) -> tuple[CloudObservation, CloudReward, bool]:
        """
        Execute one step of the environment.
        """
        # Create previous observation
        cpu_util_prev = self._calc_cpu_util(self.req_count, self.n_servers)
        lat_prev = self._calc_lat(cpu_util_prev)
        cost_hr_prev = self._calc_cost_hr(self.n_servers)
        
        obs_prev = CloudObservation(
            step=self.step_count,
            n_servers=self.n_servers,
            req_count=self.req_count,
            cpu_util=cpu_util_prev,
            lat=lat_prev,
            cost_hr=cost_hr_prev,
        )
        
        # Apply action to update n_servers
        if action.action_type == 0:  # Hold
            pass
        elif action.action_type == 1:  # Up
            self.n_servers = min(self.max_servers, self.n_servers + action.delta)
        elif action.action_type == 2:  # Down
            self.n_servers = max(self.min_servers, self.n_servers - action.delta)
        
        # Generate new traffic
        self.req_count = self._gen_traffic()
        
        # Calculate new state
        cpu_util = self._calc_cpu_util(self.req_count, self.n_servers)
        lat = self._calc_lat(cpu_util)
        cost_hr = self._calc_cost_hr(self.n_servers)
        
        obs_curr = CloudObservation(
            step=self.step_count,
            n_servers=self.n_servers,
            req_count=self.req_count,
            cpu_util=cpu_util,
            lat=lat,
            cost_hr=cost_hr,
        )
        
        # Calculate reward
        rew = self._calc_reward(obs_prev, obs_curr, cost_hr)
        
        # Check if done
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        return obs_curr, rew, done
    
    def reset(self) -> CloudObservation:
        """Reset environment to initial state."""
        self.step_count = 0
        self.n_servers = self.init_servers
        self.req_count = self._gen_traffic()
        
        cpu_util = self._calc_cpu_util(self.req_count, self.n_servers)
        lat = self._calc_lat(cpu_util)
        cost_hr = self._calc_cost_hr(self.n_servers)
        
        return CloudObservation(
            step=self.step_count,
            n_servers=self.n_servers,
            req_count=self.req_count,
            cpu_util=cpu_util,
            lat=lat,
            cost_hr=cost_hr,
        )
