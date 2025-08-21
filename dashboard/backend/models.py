from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict
from enum import Enum


class SenderType(str, Enum):
    FAN = "fan"
    MODEL = "model"
    CHATTER = "chatter"


class MoodType(str, Enum):
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    NEUTRAL = "neutral"
    ANGRY = "angry"
    LOVING = "loving"


class PurchaseReadiness(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SpendingFrequency(str, Enum):
    TW = "TW"  # Time-Waster - $0
    NW = "NW"  # Not Worth It - < threshold
    IO = "IO"  # In and Out - burst + inactive
    WS = "WS"  # Weekly Spender - 1-2x/week
    AD = "AD"  # Almost Daily - 4-6x/week
    ED = "ED"  # Every Day - daily


class EngagementBehavior(str, Enum):
    L = "L"    # Lurker - almost no messages
    CB = "CB"  # Chatty Buyer - frequent + spends
    CN = "CN"  # Chatty Non-Buyer - frequent, no spend
    Q = "Q"    # Quick Responder - <2 min response
    ER = "ER"  # Event-Driven - active after promos


class EmotionalAttachment(str, Enum):
    R = "R"    # Romantic - flirty, affectionate
    F = "F"    # Friendly - casual, no romance
    C = "C"    # Collector - completes sets
    E = "E"    # Escape-Seeker - stress relief
    G = "G"    # Gift-Giver - generous tipper


class LifecycleStage(str, Enum):
    N = "N"    # New - <2 weeks
    W = "W"    # Warming Up - increased engagement
    P = "P"    # Peak - stable high spend
    CO = "CO"  # Cooling Off - reduced spend
    CH = "CH"  # Churned - no engagement >30 days


class LTVSegment(str, Enum):
    W = "W"    # Whale - Top tier spenders ($5000+)
    D = "D"    # Dolphin - High value players ($1000-4999)
    S = "S"    # Shark - Strong spenders ($500-999)
    F = "F"    # Fish - Regular spenders ($100-499)
    M = "M"    # Minnow - Small spenders (<$100)


class SegmentationConfig(BaseModel):
    min_profitable_threshold: float = 50.0
    gift_giver_tip_ratio: float = 0.3
    gift_giver_min_tips: float = 100.0
    emotion_min_confidence: float = 0.7
    burst_window_hours: int = 48
    inactivity_days: int = 14
    churn_days: int = 30
    timezone: str = "Asia/Manila"
    
    # LTV segment thresholds
    ltv_whale_threshold: float = 5000.0
    ltv_dolphin_threshold: float = 1000.0
    ltv_shark_threshold: float = 500.0
    ltv_fish_threshold: float = 100.0
    version: int = 1


class FanSegment(BaseModel):
    spending_frequency: SpendingFrequency
    engagement_behavior: EngagementBehavior
    emotional_attachment: EmotionalAttachment
    lifecycle_stage: LifecycleStage
    ltv_segment: Optional[LTVSegment] = None
    segment_code: str
    is_provisional: bool = False
    last_change_at: Optional[datetime] = None
    change_reason: Optional[str] = None
    computation_features: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    datetime: datetime
    sender_type: SenderType
    sender_id: str
    message: str
    price: float = 0.0
    purchased: bool = False
    tips: float = 0.0
    revenue: float = 0.0
    conversation_id: str
    emotions: Optional[Dict[str, float]] = None
    dominant_emotion: Optional[str] = None


class FanSummary(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    fan_id: str
    model_name: str
    total_spent: float
    days_since_last_purchase: Optional[int]
    last_message: Optional[str]
    last_message_time: Optional[datetime]
    response_pattern: str
    conversation_streak: int
    current_mood: MoodType
    mood_emoji: str
    purchase_readiness: PurchaseReadiness
    purchase_readiness_score: int
    best_time_to_chat: str
    successful_topics: List[str]
    preferred_message_style: str
    next_action_suggestion: str
    recent_topics: List[str]
    personal_notes: str = ""
    message_count: int
    avg_response_time_minutes: Optional[float]
    total_conversations: int
    # Segmentation fields
    segmentation: Optional[FanSegment] = None


class FanHistory(BaseModel):
    fan_id: str
    messages: List[Message]
    total_spent: float
    first_interaction: datetime
    last_interaction: datetime
    purchase_history: List[Dict[str, Any]]


class DashboardStats(BaseModel):
    total_fans: int
    active_fans_today: int
    total_revenue: float
    avg_fan_value: float
    top_spenders: List[Dict[str, Any]]
    mood_distribution: Dict[MoodType, int]