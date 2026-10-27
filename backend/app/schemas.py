from typing import Optional

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    case_study: Optional[str] = Field(None, description="'dotcom' | 'gfc' | 'covid' | None for custom")
    assets: Optional[list[str]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    primary_symbol: Optional[str] = None
    rebalance_freq: int = 21
    cost_bps: float = 5.0
    es_generations: int = 40
    es_population: int = 32
