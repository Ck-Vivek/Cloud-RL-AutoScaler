from pydantic import BaseModel, Field, model_validator


class CloudObservation(BaseModel):
    step: int = Field(ge=0)
    n_servers: int = Field(ge=1)
    req_count: int = Field(ge=0)
    # Allow burstable CPU observations while env can still cap dashboard values.
    cpu_util: float = Field(ge=0.0, le=200.0)
    lat: float = Field(ge=0.0)
    cost_hr: float = Field(ge=0.0)


class CloudAction(BaseModel):
    action_type: int
    delta: int = Field(default=1, ge=0)
    target_servers: int | None = Field(default=None, ge=1)
    note: str | None = None

    @model_validator(mode="after")
    def validate_action(self) -> "CloudAction":
        if self.action_type not in (0, 1, 2):
            raise ValueError("action_type must be 0 (Hold), 1 (Up), or 2 (Down)")

        if self.action_type == 0:
            if self.delta != 0:
                self.delta = 0
            if self.target_servers is not None:
                raise ValueError("target_servers must be None for Hold action")

        if self.action_type in (1, 2) and self.delta < 1:
            raise ValueError("delta must be >= 1 for Up/Down actions")

        return self


class CloudReward(BaseModel):
    rew: float
    cost_pen: float = 0.0
    lat_pen: float = 0.0
    act_pen: float = 0.0


class CloudEpisodeSummary(BaseModel):
    score: float
    total_cost: float = Field(ge=0.0)
    tokens_used: int = Field(ge=0)
    sla_breaches: int = Field(ge=0)
    n_steps: int = Field(ge=0)
