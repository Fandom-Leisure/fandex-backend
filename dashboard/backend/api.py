from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Optional, List
from datetime import datetime
import logging
import json
from pathlib import Path

from .data_processor import DataProcessor
from .fan_analyzer import FanAnalyzer
from .models import FanSummary, FanHistory, DashboardStats, SegmentationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fan Intelligence Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

data_processor = DataProcessor()
fan_analyzer = FanAnalyzer(data_processor)

@app.get("/")
async def read_index():
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/api/health")
async def health_check():
    """Check if API is running."""
    return {"status": "healthy", "timestamp": datetime.now()}


@app.get("/api/models")
async def get_models():
    """Get list of all unique model names."""
    try:
        models = data_processor.merged_df['model_name'].unique().tolist()
        models = sorted([m for m in models if m])  # Remove None values and sort
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fans", response_model=List[dict])
async def get_fans(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = None,
    model_name: Optional[str] = None,
    group_by_model: bool = Query(False, description="Group by fan-model pairs instead of just fans"),
    spending_frequency: Optional[str] = None,
    engagement_behavior: Optional[str] = None,
    emotional_attachment: Optional[str] = None,
    lifecycle_stage: Optional[str] = None,
    ltv_segment: Optional[str] = None
):
    """Get list of fans with basic info, optionally grouped by fan-model pairs."""
    try:
        df = data_processor.merged_df
        
        if group_by_model:
            # Group by fan-model pairs
            unique_fans = df.groupby(['fan_id', 'model_name']).agg({
                'datetime': 'max',
                'revenue': 'sum',
                'tips': 'sum',
                'message': 'count'
            }).reset_index()
            unique_fans.rename(columns={'message': 'message_count'}, inplace=True)
        else:
            # Group by fan only (original behavior)
            unique_fans = df.groupby('fan_id').agg({
                'model_name': lambda x: list(x.unique()),  # Get all models for this fan
                'datetime': 'max',
                'revenue': 'sum',
                'tips': 'sum',
                'message': 'count'
            }).reset_index()
            unique_fans.rename(columns={'message': 'message_count'}, inplace=True)
        
        unique_fans['total_spent'] = unique_fans['revenue'] + unique_fans['tips']
        unique_fans = unique_fans.sort_values('datetime', ascending=False)
        
        # Apply filters
        if search:
            unique_fans = unique_fans[unique_fans['fan_id'].str.contains(search, case=False)]
        
        if model_name and group_by_model:
            unique_fans = unique_fans[unique_fans['model_name'] == model_name]
        
        # Parse segment filters
        segment_filters = {}
        if spending_frequency:
            segment_filters['spending_frequency'] = spending_frequency.split(',')
        if engagement_behavior:
            segment_filters['engagement_behavior'] = engagement_behavior.split(',')
        if emotional_attachment:
            segment_filters['emotional_attachment'] = emotional_attachment.split(',')
        if lifecycle_stage:
            segment_filters['lifecycle_stage'] = lifecycle_stage.split(',')
        if ltv_segment:
            segment_filters['ltv_segment'] = ltv_segment.split(',')
        
        result = []
        filtered_count = 0
        
        # Process fans with segment filtering
        for _, row in unique_fans.iterrows():
            # Skip if we've already collected enough results
            if len(result) >= limit:
                break
                
            # Skip if before offset
            if filtered_count < offset:
                filtered_count += 1
                continue
            
            # Get segmentation data from history
            import sqlite3
            conn = sqlite3.connect(fan_analyzer.db_path)
            cursor = conn.cursor()
            
            model_filter = row['model_name'] if group_by_model else None
            cursor.execute("""
                SELECT segment_code, last_change_at, reason, is_provisional,
                       spending_frequency, engagement_behavior, emotional_attachment, 
                       lifecycle_stage, ltv_segment
                FROM (
                    SELECT segment_code, valid_from as last_change_at, reason, is_provisional,
                           spending_frequency, engagement_behavior, emotional_attachment,
                           lifecycle_stage, ltv_segment
                    FROM fan_segment_history 
                    WHERE fan_id = ? AND model_name = ? AND valid_to IS NULL
                    ORDER BY valid_from DESC LIMIT 1
                )
            """, (row['fan_id'], model_filter or ''))
            
            seg_data = cursor.fetchone()
            conn.close()
            
            # Apply segment filters if data exists
            if seg_data and segment_filters:
                skip = False
                seg_dict = {
                    'spending_frequency': seg_data[4],
                    'engagement_behavior': seg_data[5],
                    'emotional_attachment': seg_data[6],
                    'lifecycle_stage': seg_data[7],
                    'ltv_segment': seg_data[8]
                }
                
                for field, values in segment_filters.items():
                    if seg_dict.get(field) not in values:
                        skip = True
                        break
                
                if skip:
                    continue
            elif segment_filters and not seg_data:
                # Skip fans without segmentation if filters are applied
                continue
            
            filtered_count += 1
            
            base_data = {
                'fan_id': row['fan_id'],
                'last_active': row['datetime'].isoformat(),
                'total_spent': row['total_spent'],
                'message_count': row['message_count']
            }
            
            if seg_data:
                base_data.update({
                    'segment_code': seg_data[0],
                    'last_change_at': seg_data[1],
                    'change_reason': seg_data[2],
                    'is_provisional': seg_data[3]
                })
            
            if group_by_model:
                base_data['model_name'] = row['model_name']
            else:
                # For non-grouped view, show all models
                models = row['model_name'] if isinstance(row['model_name'], list) else [row['model_name']]
                base_data['model_names'] = models
                base_data['model_name'] = models[0] if models else None
                
            result.append(base_data)
        
        return result
    except Exception as e:
        logger.error(f"Error getting fans: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fans/{fan_id}", response_model=FanSummary)
async def get_fan_summary(
    fan_id: str,
    model_name: Optional[str] = None
):
    """Get comprehensive fan summary with all 8 features."""
    try:
        summary = fan_analyzer.analyze_fan(fan_id, model_name)
        if not summary:
            raise HTTPException(status_code=404, detail="Fan not found")
        return summary
    except Exception as e:
        logger.error(f"Error analyzing fan {fan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fans/{fan_id}/history")
async def get_fan_history(
    fan_id: str,
    model_name: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    """Get fan's message history."""
    try:
        messages = data_processor.get_fan_messages(fan_id, model_name, limit)
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found")
        
        purchase_history = data_processor.get_fan_purchase_history(fan_id, model_name)
        df = data_processor.get_fan_data(fan_id, model_name)
        
        return {
            'fan_id': fan_id,
            'model_name': model_name,
            'messages': messages,
            'total_spent': sum(p['total'] for p in purchase_history),
            'first_interaction': df['datetime'].min() if not df.empty else None,
            'last_interaction': df['datetime'].max() if not df.empty else None,
            'purchase_history': purchase_history[:10]
        }
    except Exception as e:
        logger.error(f"Error getting history for fan {fan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fans/{fan_id}/notes")
async def update_fan_notes(
    fan_id: str,
    notes: dict
):
    """Update personal notes for a fan."""
    try:
        fan_analyzer.save_personal_notes(fan_id, notes.get('notes', ''))
        return {"status": "success", "fan_id": fan_id}
    except Exception as e:
        logger.error(f"Error updating notes for fan {fan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get overall dashboard statistics."""
    try:
        return fan_analyzer.get_dashboard_stats()
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fans/{fan_id}/conversations")
async def get_fan_conversations(
    fan_id: str,
    model_name: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50)
):
    """Get recent conversations for a fan."""
    try:
        df = data_processor.get_fan_data(fan_id, model_name)
        if df.empty:
            raise HTTPException(status_code=404, detail="Fan not found")
        
        conversations = df.groupby('conversation_id').agg({
            'datetime': ['min', 'max'],
            'message': 'count'
        }).reset_index()
        
        conversations.columns = ['conversation_id', 'start_time', 'end_time', 'message_count']
        conversations = conversations.sort_values('end_time', ascending=False).head(limit)
        
        result = []
        for _, conv in conversations.iterrows():
            conv_messages = df[df['conversation_id'] == conv['conversation_id']]
            preview = conv_messages.iloc[0]['message'][:100] if not conv_messages.empty else ""
            
            result.append({
                'conversation_id': conv['conversation_id'],
                'start_time': conv['start_time'].isoformat(),
                'end_time': conv['end_time'].isoformat(),
                'message_count': conv['message_count'],
                'preview': preview
            })
        
        return result
    except Exception as e:
        logger.error(f"Error getting conversations for fan {fan_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/active-fans")
async def get_active_fans(hours: int = Query(24, ge=1, le=168)):
    """Get fans active in the last N hours."""
    try:
        active_fan_ids = data_processor.get_active_fans(hours)
        
        result = []
        for fan_id in active_fan_ids[:50]:
            summary = fan_analyzer.analyze_fan(fan_id)
            if summary:
                result.append({
                    'fan_id': fan_id,
                    'model_name': summary.model_name,
                    'last_message_time': summary.last_message_time.isoformat() if summary.last_message_time else None,
                    'mood': summary.current_mood,
                    'mood_emoji': summary.mood_emoji,
                    'purchase_readiness': summary.purchase_readiness
                })
        
        return sorted(result, key=lambda x: x['last_message_time'], reverse=True)
    except Exception as e:
        logger.error(f"Error getting active fans: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/segment-changes")
async def get_segment_changes(
    since: datetime = Query(..., description="Get changes since this timestamp"),
    model_name: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get all segment changes since timestamp."""
    try:
        import sqlite3
        conn = sqlite3.connect(fan_analyzer.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT fan_id, model_name, segment_code, 
                   spending_frequency, engagement_behavior,
                   emotional_attachment, lifecycle_stage,
                   valid_from, valid_to, reason, features
            FROM fan_segment_history
            WHERE valid_from > ?
        """
        params = [since]
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
            
        query += " ORDER BY valid_from DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        changes = []
        for row in cursor.fetchall():
            changes.append({
                "fan_id": row[0],
                "model_name": row[1],
                "segment_code": row[2],
                "segments": {
                    "spending": row[3],
                    "engagement": row[4],
                    "emotional": row[5],
                    "lifecycle": row[6]
                },
                "valid_from": row[7],
                "valid_to": row[8],
                "reason": row[9],
                "features": json.loads(row[10]) if row[10] else {}
            })
        
        conn.close()
        return {"changes": changes, "count": len(changes)}
    except Exception as e:
        logger.error(f"Error getting segment changes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/segmentation-config")
async def get_segmentation_config(creator_id: str = Query("default")):
    """Get current segmentation configuration."""
    try:
        config = fan_analyzer._get_segmentation_config(creator_id)
        return config
    except Exception as e:
        logger.error(f"Error getting segmentation config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/segmentation-config")
async def update_segmentation_config(
    creator_id: str,
    config: SegmentationConfig
):
    """Create new segmentation config version."""
    try:
        import sqlite3
        conn = sqlite3.connect(fan_analyzer.db_path)
        cursor = conn.cursor()
        
        # Get current version
        cursor.execute("""
            SELECT MAX(version) FROM segmentation_config 
            WHERE creator_id = ?
        """, (creator_id,))
        
        current_version = cursor.fetchone()[0] or 0
        new_version = current_version + 1
        config.version = new_version
        
        # Insert new config
        cursor.execute("""
            INSERT INTO segmentation_config 
            (creator_id, version, config, timezone)
            VALUES (?, ?, ?, ?)
        """, (creator_id, new_version, json.dumps(config.dict()), config.timezone))
        
        conn.commit()
        conn.close()
        
        return {"status": "success", "version": new_version}
    except Exception as e:
        logger.error(f"Error updating segmentation config: {e}")
        raise HTTPException(status_code=500, detail=str(e))