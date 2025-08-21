import pickle
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from functools import lru_cache

from .models import Message, SenderType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.emotions_path = self.data_dir / "messages_with_emotions.pkl"
        self._emotions_df = None
        
        # Log the paths being used
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Emotions path: {self.emotions_path}")
        
        # Check if files exist
        if not self.emotions_path.exists():
            logger.error(f"Emotions file not found: {self.emotions_path}")
        
    @property
    def emotions_df(self) -> pd.DataFrame:
        if self._emotions_df is None:
            logger.info("Loading emotions data...")
            with open(self.emotions_path, 'rb') as f:
                self._emotions_df = pickle.load(f)
            logger.info(f"Loaded {len(self._emotions_df):,} messages with emotions")
        return self._emotions_df
    
    @property
    def merged_df(self) -> pd.DataFrame:
        # Return emotions_df which contains all the data we need
        return self.emotions_df
    
    @lru_cache(maxsize=1000)
    def get_fan_data(self, fan_id: str, model_name: Optional[str] = None) -> pd.DataFrame:
        """Get all messages for a specific fan, optionally filtered by model."""
        df = self.merged_df[self.merged_df['fan_id'] == fan_id]
        if model_name:
            df = df[df['model_name'] == model_name]
        return df.sort_values('datetime')
    
    def get_fan_messages(self, fan_id: str, model_name: Optional[str] = None, 
                        limit: int = None) -> List[Message]:
        """Get messages for a fan as Message objects."""
        df = self.get_fan_data(fan_id, model_name)
        if limit:
            df = df.tail(limit)
        
        messages = []
        for _, row in df.iterrows():
            emotions = {}
            dominant_emotion = None
            
            for col in df.columns:
                if col.startswith('emotion_') and not col.endswith('_prob'):
                    if pd.notna(row[col]):
                        emotions[col.replace('emotion_', '')] = row[col]
                elif col == 'dominant_emotion' and pd.notna(row[col]):
                    dominant_emotion = row[col]
            
            msg = Message(
                datetime=row['datetime'],
                sender_type=SenderType(row['sender_type']),
                sender_id=row['sender_id'],
                message=row['message'],
                price=row.get('price', 0.0),
                purchased=row.get('purchased', False),
                tips=row.get('tips', 0.0),
                revenue=row.get('revenue', 0.0),
                conversation_id=row['conversation_id'],
                emotions=emotions if emotions else None,
                dominant_emotion=dominant_emotion
            )
            messages.append(msg)
        
        return messages
    
    def get_fan_purchase_history(self, fan_id: str, model_name: Optional[str] = None) -> List[Dict]:
        """Get purchase history for a fan."""
        df = self.get_fan_data(fan_id, model_name)
        purchases = df[df['purchased'] == True]
        
        history = []
        for _, row in purchases.iterrows():
            history.append({
                'datetime': row['datetime'],
                'price': row['price'],
                'message': row['message'],
                'tips': row.get('tips', 0.0),
                'total': row['price'] + row.get('tips', 0.0)
            })
        
        return sorted(history, key=lambda x: x['datetime'], reverse=True)
    
    def get_fan_conversation_stats(self, fan_id: str, model_name: Optional[str] = None) -> Dict:
        """Get conversation statistics for a fan."""
        df = self.get_fan_data(fan_id, model_name)
        if df.empty:
            return {}
        
        conversations = df.groupby('conversation_id').agg({
            'datetime': ['min', 'max'],
            'message': 'count'
        })
        
        total_convs = len(conversations)
        avg_messages_per_conv = conversations[('message', 'count')].mean()
        
        df['time_diff'] = df['datetime'].diff()
        fan_messages = df[df['sender_type'] == 'fan']
        model_messages = df[df['sender_type'] != 'fan']
        
        if len(fan_messages) > 1:
            response_times = []
            for i in range(1, len(df)):
                if (df.iloc[i]['sender_type'] != 'fan' and 
                    df.iloc[i-1]['sender_type'] == 'fan'):
                    time_diff = (df.iloc[i]['datetime'] - df.iloc[i-1]['datetime']).total_seconds() / 60
                    if time_diff < 60:
                        response_times.append(time_diff)
            
            avg_response_time = sum(response_times) / len(response_times) if response_times else None
        else:
            avg_response_time = None
        
        return {
            'total_conversations': total_convs,
            'avg_messages_per_conversation': avg_messages_per_conv,
            'total_messages': len(df),
            'fan_messages': len(fan_messages),
            'model_messages': len(model_messages),
            'avg_response_time_minutes': avg_response_time
        }
    
    def get_active_fans(self, hours: int = 24) -> List[str]:
        """Get fans who were active in the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = self.merged_df[self.merged_df['datetime'] > cutoff]
        return recent['fan_id'].unique().tolist()
    
    def get_top_spenders(self, limit: int = 10) -> List[Dict]:
        """Get top spending fans."""
        fan_spending = self.merged_df.groupby('fan_id').agg({
            'revenue': 'sum',
            'tips': 'sum',
            'datetime': 'max'
        }).reset_index()
        
        fan_spending['total_spent'] = fan_spending['revenue'] + fan_spending['tips']
        top_fans = fan_spending.nlargest(limit, 'total_spent')
        
        result = []
        for _, row in top_fans.iterrows():
            result.append({
                'fan_id': row['fan_id'],
                'total_spent': row['total_spent'],
                'last_active': row['datetime']
            })
        
        return result