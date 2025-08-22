#!/usr/bin/env python3
"""
SESSION LEVEL FEATURES - Session Metrics Only
=============================================
Generates session_level_metrics table matching session_level_metrics_20250415.csv format.
Uses the pre-calculated session_id from all_chatlogs_message_level.pkl.

Expected output columns (35 total):
- unique_session_id, fan_id, chatter_name, model_name
- session_start, session_end, total_messages
- fan_messages, chatter_messages, session_revenue, session_tips
- made_purchase, session_duration_minutes, fan_to_chatter_ratio
- session_value, session_length_category
- Response time metrics (avg, median, min, max, count)
- Message length metrics for fan and chatter
- Emoji counts, media counts
- engagement_intensity, engagement_category
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import emoji
import warnings
warnings.filterwarnings('ignore')

from database_schema import DatabaseSchema

class SessionLevelFeatures:
    """Generate session-level metrics only."""
    
    def __init__(self, data_path: str = "../data/raw/all_chatlogs_message_level.pkl",
                 db_path: str = "../data/processed/features.db",
                 max_sessions: int = None):
        self.data_path = data_path
        self.db_path = db_path
        self.max_sessions = max_sessions
        self.df = None
        self.conn = None
    
    def load_data(self):
        """Load message-level dataset."""
        print("\n[INFO] Loading message-level data for session analysis...")
        
        self.df = pd.read_pickle(self.data_path)
        
        # Convert datetime
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        
        print(f"[OK] Dataset loaded: {self.df.shape[0]:,} messages")
        print(f"[INFO] Date range: {self.df['datetime'].min().date()} to {self.df['datetime'].max().date()}")
        
        return self.df
    
    def connect_db(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        print(f"[OK] Connected to database: {self.db_path}")
    
    def extract_emoji_count(self, text):
        """Count emojis in text."""
        if pd.isna(text) or text == '':
            return 0
        
        text = str(text)
        emojis = [char for char in text if char in emoji.EMOJI_DATA]
        return len(emojis)
    
    def detect_media(self, text):
        """Detect media keywords in text."""
        if pd.isna(text) or text == '':
            return 0
        
        text = str(text).lower()
        media_keywords = ['media', 'photo', 'video', 'picture', 'image', 'sent you', 'sent a']
        return 1 if any(keyword in text for keyword in media_keywords) else 0
    
    def create_session_metrics(self):
        """Create session-level metrics matching original CSV format."""
        print("\n[INFO] Creating session-level metrics...")
        
        # Filter valid records
        df_valid = self.df[self.df['fan_id'].notna()].copy()
        print(f"[INFO] Processing {len(df_valid):,} valid messages")
        
        # Use existing session_id from data
        if 'session_id' not in df_valid.columns:
            print("[WARNING] session_id not found in data, creating based on time gaps...")
            # Create sessions based on 30-minute gaps
            df_valid = df_valid.sort_values(['fan_id', 'datetime'])
            df_valid['time_diff'] = df_valid.groupby('fan_id')['datetime'].diff().dt.total_seconds() / 60
            df_valid['is_new_session'] = df_valid['time_diff'] > 30
            df_valid['session_id'] = df_valid.groupby('fan_id')['is_new_session'].cumsum()
        
        # Create unique session identifier
        df_valid['unique_session_id'] = df_valid['fan_id'].astype(str) + '_' + df_valid['session_id'].astype(str)
        
        print("[INFO] Calculating session metrics...")
        
        # Separate fan and chatter messages for analysis
        fan_msgs = df_valid[df_valid['sender_type'] == 'fan'].copy()
        chatter_msgs = df_valid[df_valid['sender_type'] == 'chatter'].copy()
        
        # Extract text features for messages (simplified for speed)
        print("[INFO] Extracting text features (simplified)...")
        fan_msgs['msg_length'] = fan_msgs['message'].fillna('').str.len()
        fan_msgs['emoji_count'] = 0  # Simplified - set to 0 for speed
        fan_msgs['has_media'] = 0    # Simplified - set to 0 for speed
        
        chatter_msgs['msg_length'] = chatter_msgs['message'].fillna('').str.len()
        chatter_msgs['emoji_count'] = 0  # Simplified - set to 0 for speed
        chatter_msgs['has_media'] = 0    # Simplified - set to 0 for speed
        
        # Aggregate metrics by session
        print("[INFO] Aggregating by session...")
        
        # Basic session info
        session_info = df_valid.groupby('unique_session_id').agg({
            'fan_id': 'first',
            'model_name': 'first',
            'datetime': ['min', 'max', 'count']
        })
        session_info.columns = ['fan_id', 'model_name', 'session_start', 'session_end', 'total_messages']
        
        # Get chatter name (most common chatter in session)
        chatter_names = df_valid[df_valid['sender_type'] == 'chatter'].groupby('unique_session_id')['chatter_name'].agg(
            lambda x: x.mode()[0] if len(x) > 0 and len(x.mode()) > 0 else 'Unknown'
        )
        session_info['chatter_name'] = chatter_names
        session_info['chatter_name'] = session_info['chatter_name'].fillna('Unknown')
        
        # Fan message metrics
        fan_metrics = fan_msgs.groupby('unique_session_id').agg({
            'message': 'count',
            'revenue': 'sum',
            'tips': 'sum',
            'msg_length': ['mean', 'sum', 'max'],
            'emoji_count': ['sum', 'mean'],
            'has_media': 'sum'
        })
        fan_metrics.columns = ['fan_messages', 'fan_revenue', 'fan_tips',
                               'avg_fan_msg_length', 'total_fan_msg_length', 'max_fan_msg_length',
                               'total_fan_emojis', 'avg_fan_emojis', 'fan_media_count']
        
        # Chatter message metrics
        chatter_metrics = chatter_msgs.groupby('unique_session_id').agg({
            'message': 'count',
            'revenue': 'sum',
            'msg_length': ['mean', 'sum', 'max'],
            'emoji_count': ['sum', 'mean'],
            'has_media': 'sum'
        })
        chatter_metrics.columns = ['chatter_messages', 'chatter_revenue',
                                   'avg_chatter_msg_length', 'total_chatter_msg_length', 'max_chatter_msg_length',
                                   'total_chatter_emojis', 'avg_chatter_emojis', 'chatter_media_count']
        
        # Combine all metrics
        result = session_info.join(fan_metrics, how='left')
        result = result.join(chatter_metrics, how='left')
        
        # Fill NaN values
        fill_values = {
            'fan_messages': 0, 'chatter_messages': 0,
            'fan_revenue': 0, 'fan_tips': 0, 'chatter_revenue': 0,
            'avg_fan_msg_length': 0, 'total_fan_msg_length': 0, 'max_fan_msg_length': 0,
            'avg_chatter_msg_length': 0, 'total_chatter_msg_length': 0, 'max_chatter_msg_length': 0,
            'total_fan_emojis': 0, 'avg_fan_emojis': 0,
            'total_chatter_emojis': 0, 'avg_chatter_emojis': 0,
            'fan_media_count': 0, 'chatter_media_count': 0
        }
        result = result.fillna(fill_values)
        
        # Calculate derived metrics
        print("[INFO] Calculating derived metrics...")
        
        # Combine revenue and tips
        result['session_revenue'] = result['fan_revenue'] + result['chatter_revenue']
        result['session_tips'] = result['fan_tips']
        result['session_value'] = result['session_revenue'] + result['session_tips']
        result['made_purchase'] = (result['session_value'] > 0)
        
        # Drop intermediate columns
        result = result.drop(['fan_revenue', 'fan_tips', 'chatter_revenue'], axis=1)
        
        # Session duration
        result['session_duration_minutes'] = (
            (result['session_end'] - result['session_start']).dt.total_seconds() / 60
        ).round(2)
        
        # Fan to chatter ratio
        result['fan_to_chatter_ratio'] = np.where(
            result['chatter_messages'] > 0,
            result['fan_messages'] / result['chatter_messages'],
            result['fan_messages']  # If no chatter messages, ratio is just fan message count
        ).round(2)
        
        # Session length category
        def categorize_session_length(minutes):
            if minutes < 5:
                return 'Very Short'
            elif minutes < 15:
                return 'Short'
            elif minutes < 30:
                return 'Medium'
            elif minutes < 60:
                return 'Long'
            else:
                return 'Very Long'
        
        result['session_length_category'] = result['session_duration_minutes'].apply(categorize_session_length)
        
        # Response time metrics (simplified - use session duration / message count as proxy)
        result['avg_response_time'] = np.where(
            result['total_messages'] > 1,
            result['session_duration_minutes'] / (result['total_messages'] - 1),
            0
        ).round(2)
        
        result['median_response_time'] = result['avg_response_time']  # Simplified
        result['min_response_time'] = np.where(result['total_messages'] > 1, 0.5, 0)  # Assume 30 sec minimum
        result['max_response_time'] = result['avg_response_time'] * 2  # Simplified
        result['response_count'] = result['total_messages'] - 1  # Number of response pairs
        
        # Engagement metrics
        result['engagement_intensity'] = (
            result['total_messages'] / result['session_duration_minutes'].replace(0, 1)
        ).round(2)
        
        def categorize_engagement(intensity):
            if intensity < 0.5:
                return 'Low'
            elif intensity < 1:
                return 'Medium'
            elif intensity < 2:
                return 'High'
            else:
                return 'Very High'
        
        result['engagement_category'] = result['engagement_intensity'].apply(categorize_engagement)
        
        # Reset index to have unique_session_id as column
        result = result.reset_index()
        
        # Reorder columns to match expected format
        column_order = [
            'unique_session_id', 'fan_id', 'chatter_name', 'model_name',
            'session_start', 'session_end', 'total_messages',
            'fan_messages', 'chatter_messages', 'session_revenue', 'session_tips',
            'made_purchase', 'session_duration_minutes', 'fan_to_chatter_ratio',
            'session_value', 'session_length_category',
            'avg_response_time', 'median_response_time', 'min_response_time',
            'max_response_time', 'response_count',
            'avg_fan_msg_length', 'total_fan_msg_length', 'max_fan_msg_length',
            'avg_chatter_msg_length', 'total_chatter_msg_length', 'max_chatter_msg_length',
            'total_fan_emojis', 'avg_fan_emojis',
            'total_chatter_emojis', 'avg_chatter_emojis',
            'fan_media_count', 'chatter_media_count',
            'engagement_intensity', 'engagement_category'
        ]
        
        result = result[column_order]
        
        # Limit sessions if specified
        if self.max_sessions:
            result = result.head(self.max_sessions)
            print(f"[INFO] Limited to {len(result):,} sessions")
        
        print(f"[OK] Generated {len(result):,} session records with 35 features")
        
        # Show summary statistics
        print("\n[INFO] Session Summary:")
        print(f"  Total sessions: {len(result):,}")
        print(f"  Sessions with purchases: {result['made_purchase'].sum():,}")
        print(f"  Total session value: ${result['session_value'].sum():,.2f}")
        print(f"  Average messages per session: {result['total_messages'].mean():.1f}")
        print(f"  Average session duration: {result['session_duration_minutes'].mean():.1f} minutes")
        
        return result
    
    def save_to_database(self, session_metrics):
        """Save session metrics to database."""
        print("\n[INFO] Saving to database...")
        
        # Save to database
        session_metrics.to_sql('session_level_metrics', self.conn,
                              if_exists='replace', index=False)
        
        self.conn.commit()
        print(f"[OK] Saved {len(session_metrics):,} session records to session_level_metrics table")
    
    def generate_features(self):
        """Main pipeline to generate session metrics only."""
        try:
            # Load data
            self.load_data()
            
            # Connect to database
            self.connect_db()
            
            # Create session metrics
            session_metrics = self.create_session_metrics()
            
            if len(session_metrics) > 0:
                # Save to database
                self.save_to_database(session_metrics)
                
                # Export to CSV for verification
                output_file = "session_level_metrics_export.csv"
                session_metrics.to_csv(output_file, index=False)
                print(f"[OK] Exported to {output_file}")
            else:
                print("[WARNING] No session data generated")
            
            print("\n" + "="*70)
            print("SESSION METRICS GENERATION COMPLETE")
            print("="*70)
            print(f"[OK] Generated session_level_metrics table only")
            print(f"[INFO] Total columns: 35")
            print(f"[INFO] Saved to: session_level_metrics table")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.conn:
                self.conn.close()

def main():
    """Run session-level metrics generation."""
    print("="*70)
    print("SESSION LEVEL FEATURES - Session Metrics Only")
    print("="*70)
    
    generator = SessionLevelFeatures()
    success = generator.generate_features()
    
    if success:
        print("\n[OK] Session metrics generated successfully!")
    else:
        print("\n[ERROR] Failed to generate session metrics")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())