"""
Core data structures for CrowdVision system
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import numpy as np


@dataclass
class Detection:
    """Person detection from object detector"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    frame_id: int
    camera_id: str
    timestamp: datetime


@dataclass
class Track:
    """Multi-camera person track"""
    track_id: int
    current_bbox: Tuple[float, float, float, float]
    embedding: np.ndarray  # 512-dim Re-ID feature
    trajectory: List[Tuple[float, float]]  # Historical positions
    camera_id: str
    last_seen: datetime
    confidence: float


@dataclass
class DensityMap:
    """Crowd density estimation result"""
    zone_id: str
    density_array: np.ndarray  # 2D density map
    total_count: float
    occupancy_percent: float
    timestamp: datetime
    camera_id: str


@dataclass
class Bottleneck:
    """Predicted bottleneck event"""
    zone_id: str
    predicted_density: float
    time_to_occurrence_minutes: int
    severity: float  # 0-1 scale
    confidence: float
    timestamp: datetime


@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_type: str  # 'fire', 'smoke', 'surge', 'weapon', 'abandoned_object'
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]]
    camera_id: str
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high'
    frame_snapshot: Optional[np.ndarray]


@dataclass
class ReIDMatch:
    """Person re-identification match"""
    query_id: str
    match_track_id: int
    similarity_score: float
    camera_id: str
    timestamp: datetime
    bbox: Tuple[float, float, float, float]
    spatiotemporal_feasible: bool


@dataclass
class Sentiment:
    """Crowd sentiment analysis result"""
    zone_id: str
    sentiment_class: str  # 'calm', 'excited', 'agitated', 'panic'
    confidence: float
    visual_score: float
    audio_score: Optional[float]
    social_score: Optional[float]
    timestamp: datetime
    alert_level: str  # 'green', 'yellow', 'red'


@dataclass
class ResourceAllocation:
    """Resource allocation recommendation"""
    incident_id: str
    recommended_unit_id: str
    assignment_score: float
    estimated_response_time_minutes: float
    route_waypoints: List[Tuple[float, float]]
    avoids_bottlenecks: bool
    timestamp: datetime


@dataclass
class AnalyticsDocument:
    """Document for RAG vector database"""
    doc_id: str
    content: str  # Natural language description
    embedding: np.ndarray  # Sentence embedding
    metadata: Dict  # zone_id, timestamp, metric_type, etc.
    timestamp: datetime
