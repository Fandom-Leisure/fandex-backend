#!/usr/bin/env python3
"""
MESSAGE LEVEL FEATURES - Emoji Analysis Only
============================================
Generates emoji_analysis table with proper message-level format.
Keeps the original structure with 'message' and 'sender_type' columns.
Calculates actual emoji statistics (not zeros).

Output columns:
- fan_id, chatter_name, model_name, datetime
- message, sender_type, sender_id
- emoji_count, unique_emoji_count, emoji_diversity
- revenue, tips, purchased, total_value
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import emoji
import warnings
warnings.filterwarnings('ignore')

from database_schema import DatabaseSchema

class MessageLevelFeatures:
    """Generate emoji analysis features with proper message-level format."""
    
    def __init__(self, data_path: str = "../data/raw/all_chatlogs_message_level.pkl",
                 db_path: str = "../data/processed/features.db",
                 max_records: int = 5000000):
        self.data_path = data_path
        self.db_path = db_path
        self.max_records = max_records
        self.df = None
        self.conn = None
    
    def load_data(self):
        """Load message-level dataset."""
        print("\n[INFO] Loading message-level data for emoji analysis...")
        
        self.df = pd.read_pickle(self.data_path)
        
        # Convert datetime column
        self.df['datetime'] = pd.to_datetime(self.df['datetime'])
        
        print(f"[OK] Dataset loaded: {self.df.shape[0]:,} messages")
        print(f"[INFO] Date range: {self.df['datetime'].min().date()} to {self.df['datetime'].max().date()}")
        
        return self.df
    
    def connect_db(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        print(f"[OK] Connected to database: {self.db_path}")
    
    def extract_emoji_stats(self, text):
        """Extract emoji statistics from text."""
        if pd.isna(text) or text == '':
            return pd.Series({
                'emoji_count': 0,
                'unique_emoji_count': 0,
                'emoji_diversity': 0.0
            })
        
        # Convert to string to handle any type
        text = str(text)
        
        # Find all emojis in the text
        emojis = [char for char in text if char in emoji.EMOJI_DATA]
        
        if not emojis:
            return pd.Series({
                'emoji_count': 0,
                'unique_emoji_count': 0,
                'emoji_diversity': 0.0
            })
        
        count = len(emojis)
        unique = len(set(emojis))
        diversity = unique / count if count > 0 else 0.0
        
        return pd.Series({
            'emoji_count': count,
            'unique_emoji_count': unique,
            'emoji_diversity': round(diversity, 4)
        })
    
    def create_emoji_analysis(self):
        """Create emoji analysis keeping original message-level format."""
        print("\n[INFO] Creating emoji analysis with message-level format...")
        
        # Filter valid records
        df_valid = self.df[self.df['fan_id'].notna()].copy()
        
        # Sort by datetime to get most recent if we need to limit
        df_valid = df_valid.sort_values('datetime', ascending=False)
        
        if len(df_valid) > self.max_records:
            print(f"[INFO] Limiting to most recent {self.max_records:,} messages (from {len(df_valid):,})")
            df_valid = df_valid.head(self.max_records)
        else:
            print(f"[INFO] Processing {len(df_valid):,} messages")
        
        # Sort back to chronological order
        df_valid = df_valid.sort_values('datetime')
        
        print("[INFO] Building emoji analysis format...")
        
        # Start with the original columns
        result = pd.DataFrame()
        
        # Keep original structure columns
        result['fan_id'] = df_valid['fan_id'].values
        result['chatter_name'] = df_valid['chatter_name'].fillna('Unknown').values
        result['model_name'] = df_valid['model_name'].fillna('Unknown').values
        result['datetime'] = df_valid['datetime'].values
        
        # Keep the original message format
        result['message'] = df_valid['message'].fillna('').values
        result['sender_type'] = df_valid['sender_type'].values
        result['sender_id'] = df_valid['sender_id'].fillna('Unknown').values
        
        # Extract emoji statistics from messages
        print("[INFO] Extracting emoji features...")
        emoji_stats = df_valid['message'].apply(self.extract_emoji_stats)
        
        result['emoji_count'] = emoji_stats['emoji_count'].astype(int)
        result['unique_emoji_count'] = emoji_stats['unique_emoji_count'].astype(int)
        result['emoji_diversity'] = emoji_stats['emoji_diversity'].astype(float)
        
        # Revenue and tips - handle based on sender_type
        # Fan messages can have tips, chatter messages can have revenue
        result['revenue'] = 0.0
        result['tips'] = 0.0
        
        # Get revenue/tips from original data
        revenue_values = df_valid['revenue'].fillna(0).values
        tips_values = df_valid['tips'].fillna(0).values
        
        # Fans send tips, chatters generate revenue
        fan_mask = (df_valid['sender_type'] == 'fan').values
        chatter_mask = (df_valid['sender_type'] == 'chatter').values
        
        result.loc[fan_mask, 'tips'] = tips_values[fan_mask]
        result.loc[chatter_mask, 'revenue'] = revenue_values[chatter_mask]
        
        # Purchase information
        result['purchased'] = (result['revenue'] > 0) | (result['tips'] > 0)
        result['total_value'] = result['revenue'] + result['tips']
        
        print(f"[OK] Generated emoji analysis with {len(result):,} records")
        
        # Show summary statistics
        print("\n[INFO] Emoji Analysis Summary:")
        print(f"  Total messages: {len(result):,}")
        print(f"  Fan messages: {fan_mask.sum():,}")
        print(f"  Chatter messages: {chatter_mask.sum():,}")
        print(f"  Messages with emojis: {(result['emoji_count'] > 0).sum():,}")
        print(f"  Messages with purchases: {result['purchased'].sum():,}")
        print(f"  Total revenue (from chatters): ${result['revenue'].sum():,.2f}")
        print(f"  Total tips (from fans): ${result['tips'].sum():,.2f}")
        print(f"  Combined total: ${result['total_value'].sum():,.2f}")
        
        # Show emoji statistics
        emoji_messages = result[result['emoji_count'] > 0]
        if len(emoji_messages) > 0:
            print(f"\n[INFO] Emoji Statistics:")
            print(f"  Messages with emojis: {len(emoji_messages):,} ({len(emoji_messages)/len(result)*100:.1f}%)")
            print(f"  Average emojis per message (when present): {emoji_messages['emoji_count'].mean():.1f}")
            print(f"  Max emojis in a message: {emoji_messages['emoji_count'].max()}")
            
            # Breakdown by sender type
            fan_emojis = emoji_messages[emoji_messages['sender_type'] == 'fan']
            chatter_emojis = emoji_messages[emoji_messages['sender_type'] == 'chatter']
            print(f"  Fan messages with emojis: {len(fan_emojis):,}")
            print(f"  Chatter messages with emojis: {len(chatter_emojis):,}")
        
        return result
    
    def save_to_database(self, emoji_analysis):
        """Save emoji analysis to database in chunks."""
        print("\n[INFO] Saving to database...")
        
        # Save in chunks for better performance
        chunk_size = 100000
        total_chunks = (len(emoji_analysis) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(emoji_analysis), chunk_size):
            chunk = emoji_analysis.iloc[i:i+chunk_size]
            if i == 0:
                chunk.to_sql('emoji_analysis', self.conn, if_exists='replace', index=False)
            else:
                chunk.to_sql('emoji_analysis', self.conn, if_exists='append', index=False)
            
            chunk_num = (i // chunk_size) + 1
            print(f"  Saved chunk {chunk_num}/{total_chunks}")
        
        self.conn.commit()
        print(f"[OK] Saved {len(emoji_analysis):,} records to emoji_analysis table")
    
    def generate_features(self):
        """Main pipeline to generate emoji analysis."""
        try:
            # Load data
            self.load_data()
            
            # Connect to database
            self.connect_db()
            
            # Create emoji analysis
            emoji_analysis = self.create_emoji_analysis()
            
            if len(emoji_analysis) > 0:
                # Save to database
                self.save_to_database(emoji_analysis)
                
                # Export sample to CSV for verification
                sample_size = min(100000, len(emoji_analysis))
                output_file = "emoji_analysis_export.csv"
                emoji_analysis.head(sample_size).to_csv(output_file, index=False)
                print(f"[OK] Exported {sample_size:,} sample records to {output_file}")
                
                # Show sample of messages with emojis
                sample_with_emojis = emoji_analysis[emoji_analysis['emoji_count'] > 0].head(3)
                if len(sample_with_emojis) > 0:
                    print(f"\n[INFO] Found {len(sample_with_emojis)} sample messages with emojis")
            else:
                print("[WARNING] No emoji analysis data generated")
            
            print("\n" + "="*70)
            print("EMOJI ANALYSIS GENERATION COMPLETE")
            print("="*70)
            print(f"[OK] Generated emoji analysis table")
            print(f"[INFO] Format: message-level with sender_type")
            print(f"[INFO] Saved to: emoji_analysis table")
            
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
    """Run message-level emoji analysis generation."""
    print("="*70)
    print("MESSAGE LEVEL FEATURES - Emoji Analysis")
    print("="*70)
    
    generator = MessageLevelFeatures()
    success = generator.generate_features()
    
    if success:
        print("\n[OK] Emoji analysis generated successfully!")
    else:
        print("\n[ERROR] Failed to generate emoji analysis")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())