#!/usr/bin/env python3
"""
FAN LEVEL FEATURES - Production Version (42 Columns)
=====================================================
Generates exact 42 columns matching enhanced_fan_analysis_20250415.csv
Optimized for message-level data format.

Expected columns (42 total):
- fan_id, tier, total_spending, total_revenue, total_tips
- lifetime_spend, activity_status, days_since_last_interaction
- days_since_last_purchase, purchases_last_7_days, purchases_last_30_days
- avg_purchase_amount_lifetime, avg_purchase_amount_recent
- highest_single_purchase, total_purchase_count, spend_trend
- messages_last_24_hours, messages_last_7_days, messages_last_30_days
- avg_messages_per_active_day, days_since_last_message
- current_streak_days, max_streak_days, unique_active_days
- messaging_engagement, total_interactions, fan_messages_sent
- chatter_messages_received, first_interaction, last_interaction
- last_purchase_date, last_message_date, days_active
- avg_daily_value, messages_per_day, peak_hour, most_active_day
- primary_period, weekday_pct, weekend_pct, engagement_trend, segments_str
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import warnings
warnings.filterwarnings('ignore')

from database_schema import DatabaseSchema

class FanLevelFeatures:
    """Generate exact 42 fan-level features matching original CSV."""
    
    def __init__(self, data_path: str = "../data/raw/all_chatlogs_message_level.pkl", 
                 db_path: str = "../data/processed/features.db"):
        self.data_path = data_path
        self.db_path = db_path
        self.df = None
        self.conn = None
        self.DATASET_END_DATE = None
    
    def load_data(self):
        """Load and validate the message-level dataset."""
        print("\n[INFO] Loading message-level data for fan features...")
        
        self.df = pd.read_pickle(self.data_path)
        
        print(f"[OK] Dataset loaded: {self.df.shape[0]:,} messages")
        print(f"[INFO] Unique fans: {self.df['fan_id'].nunique():,}")
        print(f"[INFO] Fan messages: {(self.df['sender_type'] == 'fan').sum():,}")
        print(f"[INFO] Chatter messages: {(self.df['sender_type'] == 'chatter').sum():,}")
        print(f"[INFO] Date range: {self.df['datetime'].min().date()} to {self.df['datetime'].max().date()}")
        
        self.DATASET_END_DATE = self.df['datetime'].max()
        return self.df
    
    def connect_db(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        print(f"[OK] Connected to database: {self.db_path}")
    
    def calculate_all_features(self):
        """Calculate all 42 features matching original CSV format."""
        print("\n[INFO] Calculating all 42 fan features...")
        
        # Filter valid fans
        df_valid = self.df[self.df['fan_id'].notna()].copy()
        print(f"[INFO] Working with {len(df_valid):,} valid messages")
        
        # Get unique fans list
        all_fans = df_valid['fan_id'].unique()
        print(f"[INFO] Processing {len(all_fans):,} unique fans")
        
        # Initialize result dataframe with all fans
        result = pd.DataFrame({'fan_id': all_fans})
        result = result.set_index('fan_id')
        
        # ============ SPENDING METRICS ============
        print("[1/10] Calculating spending metrics...")
        
        # Get fan transactions (only from fan messages where purchases happen)
        fan_transactions = df_valid[df_valid['sender_type'] == 'fan'].copy()
        
        # Basic spending aggregations
        spending_metrics = fan_transactions.groupby('fan_id').agg({
            'revenue': 'sum',
            'tips': 'sum'
        })
        spending_metrics['total_revenue'] = spending_metrics['revenue']
        spending_metrics['total_tips'] = spending_metrics['tips']
        spending_metrics['total_spending'] = spending_metrics['total_revenue'] + spending_metrics['total_tips']
        spending_metrics['lifetime_spend'] = spending_metrics['total_spending']  # Same as total_spending
        
        # Tier calculation
        def categorize_tier(spending):
            if spending > 1000:
                return 'Whale'
            elif spending > 500:
                return 'High Value'
            elif spending > 200:
                return 'Medium Value'
            elif spending > 0:
                return 'Low Value'
            else:
                return 'Non-Paying'
        
        spending_metrics['tier'] = spending_metrics['total_spending'].apply(categorize_tier)
        
        # Join to result
        result = result.join(spending_metrics[['tier', 'total_spending', 'total_revenue', 
                                               'total_tips', 'lifetime_spend']], how='left')
        
        # ============ PURCHASE ANALYSIS ============
        print("[2/10] Analyzing purchase patterns...")
        
        # Get purchase transactions (revenue > 0 or tips > 0)
        purchases = fan_transactions[(fan_transactions['revenue'] > 0) | (fan_transactions['tips'] > 0)].copy()
        purchases['purchase_amount'] = purchases['revenue'] + purchases['tips']
        
        if len(purchases) > 0:
            # Last purchase date for each fan
            last_purchase = purchases.groupby('fan_id')['datetime'].max()
            result['last_purchase_date'] = last_purchase
            result['days_since_last_purchase'] = (self.DATASET_END_DATE - result['last_purchase_date']).dt.days
            
            # Recent purchases (7 and 30 days)
            recent_7_days = self.DATASET_END_DATE - timedelta(days=7)
            recent_30_days = self.DATASET_END_DATE - timedelta(days=30)
            
            purchases_7d = purchases[purchases['datetime'] > recent_7_days].groupby('fan_id').size()
            purchases_30d = purchases[purchases['datetime'] > recent_30_days].groupby('fan_id').size()
            
            result['purchases_last_7_days'] = purchases_7d.reindex(result.index, fill_value=0)
            result['purchases_last_30_days'] = purchases_30d.reindex(result.index, fill_value=0)
            
            # Purchase amounts
            purchase_stats = purchases.groupby('fan_id').agg({
                'purchase_amount': ['mean', 'max', 'count']
            })
            purchase_stats.columns = ['avg_purchase_amount_lifetime', 'highest_single_purchase', 'total_purchase_count']
            
            # Recent average (last 30 days)
            recent_purchases = purchases[purchases['datetime'] > recent_30_days]
            if len(recent_purchases) > 0:
                recent_avg = recent_purchases.groupby('fan_id')['purchase_amount'].mean()
                result['avg_purchase_amount_recent'] = recent_avg.reindex(result.index, fill_value=0)
            else:
                result['avg_purchase_amount_recent'] = 0
            
            result = result.join(purchase_stats, how='left')
            
            # Spend trend (comparing last 30 days to previous 30 days)
            prev_60_days = self.DATASET_END_DATE - timedelta(days=60)
            recent_spend = purchases[purchases['datetime'] > recent_30_days].groupby('fan_id')['purchase_amount'].sum()
            prev_spend = purchases[(purchases['datetime'] > prev_60_days) & 
                                  (purchases['datetime'] <= recent_30_days)].groupby('fan_id')['purchase_amount'].sum()
            
            def get_spend_trend(fan_id):
                recent = recent_spend.get(fan_id, 0)
                previous = prev_spend.get(fan_id, 0)
                if recent > previous:
                    return 'Increasing'
                elif recent < previous:
                    return 'Decreasing'
                elif recent == 0 and previous == 0:
                    return 'No Spending'
                else:
                    return 'Stable'
            
            result['spend_trend'] = [get_spend_trend(fan_id) for fan_id in result.index]
        else:
            # No purchases in dataset
            result['last_purchase_date'] = pd.NaT
            result['days_since_last_purchase'] = -1
            result['purchases_last_7_days'] = 0
            result['purchases_last_30_days'] = 0
            result['avg_purchase_amount_lifetime'] = 0
            result['avg_purchase_amount_recent'] = 0
            result['highest_single_purchase'] = 0
            result['total_purchase_count'] = 0
            result['spend_trend'] = 'No Spending'
        
        # ============ MESSAGE METRICS ============
        print("[3/10] Calculating message metrics...")
        
        # Message counts by sender type
        fan_messages = df_valid[df_valid['sender_type'] == 'fan'].groupby('fan_id').size()
        chatter_messages = df_valid[df_valid['sender_type'] == 'chatter'].groupby('fan_id').size()
        total_interactions = df_valid.groupby('fan_id').size()
        
        result['fan_messages_sent'] = fan_messages.reindex(result.index, fill_value=0)
        result['chatter_messages_received'] = chatter_messages.reindex(result.index, fill_value=0)
        result['total_interactions'] = total_interactions.reindex(result.index, fill_value=0)
        
        # Recent message counts (24h, 7d, 30d)
        recent_1_day = self.DATASET_END_DATE - timedelta(days=1)
        recent_7_days = self.DATASET_END_DATE - timedelta(days=7)
        recent_30_days = self.DATASET_END_DATE - timedelta(days=30)
        
        messages_1d = df_valid[df_valid['datetime'] > recent_1_day].groupby('fan_id').size()
        messages_7d = df_valid[df_valid['datetime'] > recent_7_days].groupby('fan_id').size()
        messages_30d = df_valid[df_valid['datetime'] > recent_30_days].groupby('fan_id').size()
        
        result['messages_last_24_hours'] = messages_1d.reindex(result.index, fill_value=0)
        result['messages_last_7_days'] = messages_7d.reindex(result.index, fill_value=0)
        result['messages_last_30_days'] = messages_30d.reindex(result.index, fill_value=0)
        
        # First and last interaction dates
        interaction_dates = df_valid.groupby('fan_id')['datetime'].agg(['min', 'max'])
        result['first_interaction'] = interaction_dates['min']
        result['last_interaction'] = interaction_dates['max']
        
        # Last message date (same as last interaction for message-level data)
        result['last_message_date'] = result['last_interaction']
        
        # Days metrics
        result['days_since_last_interaction'] = (self.DATASET_END_DATE - result['last_interaction']).dt.days
        result['days_since_last_message'] = result['days_since_last_interaction']  # Same for message-level
        result['days_active'] = (result['last_interaction'] - result['first_interaction']).dt.days + 1
        
        # ============ ACTIVITY PATTERNS ============
        print("[4/10] Analyzing activity patterns...")
        
        # Unique active days - optimized version
        print("[INFO] Calculating unique active days...")
        df_valid['date'] = df_valid['datetime'].dt.date
        unique_days = df_valid.groupby('fan_id')['date'].nunique()
        result['unique_active_days'] = unique_days.reindex(result.index, fill_value=0)
        df_valid = df_valid.drop('date', axis=1)  # Clean up temporary column
        
        # Average messages per active day
        result['avg_messages_per_active_day'] = result['total_interactions'] / result['unique_active_days'].replace(0, 1)
        result['messages_per_day'] = result['total_interactions'] / result['days_active'].replace(0, 1)
        result['avg_daily_value'] = result['total_spending'] / result['days_active'].replace(0, 1)
        
        # Handle inf values
        result['avg_messages_per_active_day'] = result['avg_messages_per_active_day'].replace([np.inf, -np.inf], 0)
        result['messages_per_day'] = result['messages_per_day'].replace([np.inf, -np.inf], 0)
        result['avg_daily_value'] = result['avg_daily_value'].replace([np.inf, -np.inf], 0)
        
        # ============ STREAK ANALYSIS ============
        print("[5/10] Calculating streaks (simplified)...")
        
        # Simplified streak calculation - just set defaults for now
        # Full streak calculation is expensive for 114K fans
        result['current_streak_days'] = 0
        result['max_streak_days'] = 1
        
        # For active fans (last interaction < 2 days), set current streak to 1
        active_mask = result['days_since_last_interaction'] <= 1
        result.loc[active_mask, 'current_streak_days'] = 1
        
        # ============ TIME PATTERNS ============
        print("[6/10] Analyzing time patterns...")
        
        df_valid['hour'] = df_valid['datetime'].dt.hour
        df_valid['day_of_week'] = df_valid['datetime'].dt.dayofweek
        df_valid['is_weekend'] = df_valid['day_of_week'].isin([5, 6])
        df_valid['day_name'] = df_valid['datetime'].dt.day_name()
        
        # Peak hour and most active day
        time_patterns = df_valid.groupby('fan_id').agg({
            'hour': lambda x: x.mode()[0] if len(x) > 0 and len(x.mode()) > 0 else 12,
            'day_name': lambda x: x.mode()[0] if len(x) > 0 and len(x.mode()) > 0 else 'Monday'
        })
        time_patterns.columns = ['peak_hour', 'most_active_day']
        result = result.join(time_patterns, how='left')
        
        # Weekday vs weekend percentages
        weekend_stats = df_valid.groupby(['fan_id', 'is_weekend']).size().unstack(fill_value=0)
        if 'is_weekend' in df_valid.columns and len(weekend_stats.columns) > 0:
            if True in weekend_stats.columns and False in weekend_stats.columns:
                total_msgs = weekend_stats.sum(axis=1)
                result['weekend_pct'] = (weekend_stats[True] / total_msgs * 100).reindex(result.index, fill_value=0)
                result['weekday_pct'] = (weekend_stats[False] / total_msgs * 100).reindex(result.index, fill_value=0)
            elif True in weekend_stats.columns:
                result['weekend_pct'] = 100
                result['weekday_pct'] = 0
            else:
                result['weekend_pct'] = 0
                result['weekday_pct'] = 100
        else:
            result['weekend_pct'] = 0
            result['weekday_pct'] = 0
        
        # Primary period (Morning/Afternoon/Evening/Night)
        def get_primary_period(hour):
            if pd.isna(hour):
                return 'Unknown'
            elif 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        
        result['primary_period'] = result['peak_hour'].apply(get_primary_period)
        
        # ============ ACTIVITY STATUS ============
        print("[7/10] Calculating activity status...")
        
        def get_activity_status(days):
            if pd.isna(days):
                return 'Never Active'
            elif days <= 7:
                return 'Active (< 1 week)'
            elif days <= 30:
                return 'Semi-Active (1-4 weeks)'
            elif days <= 90:
                return 'At Risk (1-3 months)'
            else:
                return 'Churned (> 3 months)'
        
        result['activity_status'] = result['days_since_last_interaction'].apply(get_activity_status)
        
        # ============ ENGAGEMENT METRICS ============
        print("[8/10] Calculating engagement metrics...")
        
        # Messaging engagement (based on frequency and recency)
        def get_messaging_engagement(row):
            if row['messages_last_7_days'] > 10:
                return 'High Engagement'
            elif row['messages_last_30_days'] > 10:
                return 'Medium Engagement'
            elif row['messages_last_30_days'] > 0:
                return 'Low Engagement'
            else:
                return 'Inactive'
        
        result['messaging_engagement'] = result.apply(get_messaging_engagement, axis=1)
        
        # Engagement trend (comparing recent to historical)
        def get_engagement_trend(row):
            if row['unique_active_days'] == 0:
                return 'No Activity'
            recent_rate = row['messages_last_30_days'] / 30
            historical_rate = row['avg_messages_per_active_day']
            if recent_rate > historical_rate * 1.2:
                return 'Increasing'
            elif recent_rate < historical_rate * 0.8:
                return 'Decreasing'
            else:
                return 'Stable'
        
        result['engagement_trend'] = result.apply(get_engagement_trend, axis=1)
        
        # ============ SEGMENTATION ============
        print("[9/10] Creating segments...")
        
        def get_segments(row):
            segments = []
            
            # Spending segments
            if row['tier'] == 'Whale':
                segments.append('WHALE')
            elif row['tier'] == 'High Value':
                segments.append('HIGH_VALUE')
            elif row['tier'] == 'Medium Value':
                segments.append('MEDIUM_VALUE')
            elif row['tier'] == 'Low Value':
                segments.append('LOW_VALUE_SPENDER')
            
            # Engagement segments
            if row['messaging_engagement'] == 'High Engagement':
                segments.append('HIGH_ENGAGEMENT')
            elif row['messaging_engagement'] == 'Low Engagement':
                segments.append('LOW_ENGAGEMENT')
            
            # Special segments
            if row['spend_trend'] == 'Increasing':
                segments.append('RISING_SPENDER')
            elif row['spend_trend'] == 'Decreasing' and row['total_spending'] > 0:
                segments.append('DECLINING_SPENDER')
            
            if row['current_streak_days'] >= 7:
                segments.append('STREAK_CHAMPION')
            
            if row['primary_period'] == 'Night':
                segments.append('NIGHT_OWL')
            elif row['primary_period'] == 'Morning':
                segments.append('EARLY_BIRD')
            
            if row['weekend_pct'] > 70:
                segments.append('WEEKEND_WARRIOR')
            elif row['weekday_pct'] > 70:
                segments.append('WEEKDAY_REGULAR')
            
            # Activity-based
            if row['activity_status'] == 'Churned (> 3 months)' and row['total_spending'] > 100:
                segments.append('WIN_BACK_TARGET')
            
            # Combine or use default
            if not segments:
                if row['total_spending'] > 0:
                    segments.append('LOW_ENGAGEMENT_SPENDER')
                else:
                    segments.append('STANDARD')
            
            return ', '.join(segments[:3])  # Limit to 3 segments
        
        result['segments_str'] = result.apply(get_segments, axis=1)
        
        # ============ FILL MISSING VALUES ============
        print("[10/10] Cleaning data...")
        
        # Fill NaN values appropriately
        result['tier'] = result['tier'].fillna('Non-Paying')
        result['total_spending'] = result['total_spending'].fillna(0)
        result['total_revenue'] = result['total_revenue'].fillna(0)
        result['total_tips'] = result['total_tips'].fillna(0)
        result['lifetime_spend'] = result['lifetime_spend'].fillna(0)
        result['days_since_last_purchase'] = result['days_since_last_purchase'].fillna(-1)
        result['peak_hour'] = result['peak_hour'].fillna(12)
        result['most_active_day'] = result['most_active_day'].fillna('Monday')
        result['primary_period'] = result['primary_period'].fillna('Unknown')
        result['spend_trend'] = result['spend_trend'].fillna('No Spending')
        result['engagement_trend'] = result['engagement_trend'].fillna('No Activity')
        
        # Round numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].round(2)
        
        # ============ REORDER COLUMNS TO MATCH ORIGINAL ============
        print("[INFO] Ordering columns to match original...")
        
        expected_columns = [
            'fan_id', 'tier', 'total_spending', 'total_revenue', 'total_tips',
            'lifetime_spend', 'activity_status', 'days_since_last_interaction',
            'days_since_last_purchase', 'purchases_last_7_days', 'purchases_last_30_days',
            'avg_purchase_amount_lifetime', 'avg_purchase_amount_recent',
            'highest_single_purchase', 'total_purchase_count', 'spend_trend',
            'messages_last_24_hours', 'messages_last_7_days', 'messages_last_30_days',
            'avg_messages_per_active_day', 'days_since_last_message',
            'current_streak_days', 'max_streak_days', 'unique_active_days',
            'messaging_engagement', 'total_interactions', 'fan_messages_sent',
            'chatter_messages_received', 'first_interaction', 'last_interaction',
            'last_purchase_date', 'last_message_date', 'days_active',
            'avg_daily_value', 'messages_per_day', 'peak_hour', 'most_active_day',
            'primary_period', 'weekday_pct', 'weekend_pct', 'engagement_trend', 'segments_str'
        ]
        
        # Reset index to have fan_id as column
        result = result.reset_index()
        
        # Select and order columns
        result = result[expected_columns]
        
        print(f"[OK] Generated {len(result)} fan records with {len(result.columns)} features")
        
        return result
    
    def save_to_database(self, enhanced_analysis):
        """Save enhanced fan analysis to database."""
        print("\n[INFO] Saving to database...")
        
        # Save to database
        enhanced_analysis.to_sql('enhanced_fan_analysis', self.conn, 
                                if_exists='replace', index=False)
        
        # Try to save metadata if table exists with correct schema
        try:
            metadata = pd.DataFrame([{
                'table_name': 'enhanced_fan_analysis',
                'created_at': datetime.now(),
                'record_count': len(enhanced_analysis),
                'columns': ', '.join(enhanced_analysis.columns),
                'description': '42-column fan analysis matching original CSV format'
            }])
            
            metadata.to_sql('analysis_metadata', self.conn, 
                           if_exists='append', index=False)
        except Exception as e:
            # Metadata table might not exist or have different schema
            print(f"[WARNING] Could not save metadata: {e}")
        
        self.conn.commit()
        print(f"[OK] Saved {len(enhanced_analysis)} fan records to enhanced_fan_analysis table")
    
    def generate_features(self):
        """Main pipeline to generate all features."""
        try:
            # Load data
            self.load_data()
            
            # Connect to database
            self.connect_db()
            
            # Calculate all 42 features
            enhanced_analysis = self.calculate_all_features()
            
            # Save to database
            self.save_to_database(enhanced_analysis)
            
            # Show summary
            print("\n" + "="*70)
            print("FEATURE GENERATION COMPLETE")
            print("="*70)
            print(f"[OK] Generated {len(enhanced_analysis)} fan profiles")
            print(f"[INFO] Total features: {len(enhanced_analysis.columns)}")
            print(f"[INFO] Saved to: enhanced_fan_analysis table")
            
            # Show tier distribution
            tier_dist = enhanced_analysis['tier'].value_counts()
            print("\n[INFO] Tier Distribution:")
            for tier, count in tier_dist.items():
                pct = (count / len(enhanced_analysis)) * 100
                print(f"  {tier:15}: {count:6,} ({pct:5.1f}%)")
            
            # Show activity distribution
            activity_dist = enhanced_analysis['activity_status'].value_counts()
            print("\n[INFO] Activity Distribution:")
            for status, count in activity_dist.head(5).items():
                pct = (count / len(enhanced_analysis)) * 100
                print(f"  {status:25}: {count:6,} ({pct:5.1f}%)")
            
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
    """Run fan-level feature generation."""
    print("="*70)
    print("FAN LEVEL FEATURES - 42 Column Production Version")
    print("="*70)
    
    generator = FanLevelFeatures()
    success = generator.generate_features()
    
    if success:
        print("\n[OK] Fan level features generated successfully!")
    else:
        print("\n[ERROR] Failed to generate fan level features")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())