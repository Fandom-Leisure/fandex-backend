#!/usr/bin/env python3
"""
DATABASE SCHEMA DEFINITIONS FOR FANDOM ANALYTICS - PRODUCTION VERSION
=====================================================================
SQLite table definitions that match the exact outputs from feature generation scripts.

PRODUCTION TABLES (3 tables only):
1. enhanced_fan_analysis - 42 columns of fan-level features
2. emoji_analysis - Message-level emoji and engagement metrics
3. session_level_metrics - 35 columns of session-level features

All script analysis tables have been removed for production.
This schema exactly matches the outputs from:
- fan_level_features.py
- message_level_features.py
- session_level_features.py
"""

import sqlite3
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseSchema:
    """Manages SQLite database schema for all feature tables."""
    
    def __init__(self, db_path: str = "../data/processed/features.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        # Enable foreign keys
        self.cursor.execute("PRAGMA foreign_keys = ON")
        logger.info(f"Connected to database: {self.db_path}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def create_all_tables(self):
        """Create all feature tables."""
        self.connect()
        
        # Create tables in order
        self._create_fan_level_tables()
        self._create_message_level_tables()
        self._create_session_level_tables()
        self._create_metadata_table()
        
        self.conn.commit()
        self.close()
        logger.info("All tables created successfully")
    
    def _create_fan_level_tables(self):
        """Create fan level analysis tables - PRODUCTION VERSION."""
        
        # PRODUCTION TABLE: Enhanced fan analysis with exact 42 columns from fan_level_features.py
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_fan_analysis (
                fan_id TEXT PRIMARY KEY,
                tier TEXT,
                total_spending REAL,
                total_revenue REAL,
                total_tips REAL,
                lifetime_spend REAL,
                activity_status TEXT,
                days_since_last_interaction INTEGER,
                days_since_last_purchase INTEGER,
                purchases_last_7_days INTEGER,
                purchases_last_30_days INTEGER,
                avg_purchase_amount_lifetime REAL,
                avg_purchase_amount_recent REAL,
                highest_single_purchase REAL,
                total_purchase_count INTEGER,
                spend_trend TEXT,
                messages_last_24_hours INTEGER,
                messages_last_7_days INTEGER,
                messages_last_30_days INTEGER,
                avg_messages_per_active_day REAL,
                days_since_last_message INTEGER,
                current_streak_days INTEGER,
                max_streak_days INTEGER,
                unique_active_days INTEGER,
                messaging_engagement TEXT,
                total_interactions INTEGER,
                fan_messages_sent INTEGER,
                chatter_messages_received INTEGER,
                first_interaction TIMESTAMP,
                last_interaction TIMESTAMP,
                last_purchase_date TIMESTAMP,
                last_message_date TIMESTAMP,
                days_active INTEGER,
                avg_daily_value REAL,
                messages_per_day REAL,
                peak_hour INTEGER,
                most_active_day TEXT,
                primary_period TEXT,
                weekday_pct REAL,
                weekend_pct REAL,
                engagement_trend TEXT,
                segments_str TEXT
            )
        ''')
        
        logger.info("Created fan level table: enhanced_fan_analysis (42 columns)")
    
    def _create_message_level_tables(self):
        """Create message level analysis tables."""
        
        # PRODUCTION TABLE: Emoji analysis with exact columns from message_level_features.py
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS emoji_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fan_id TEXT,
                chatter_name TEXT,
                model_name TEXT,
                datetime TIMESTAMP,
                message TEXT,
                sender_type TEXT,
                sender_id TEXT,
                emoji_count INTEGER,
                unique_emoji_count INTEGER,
                emoji_diversity REAL,
                revenue REAL,
                tips REAL,
                purchased BOOLEAN,
                total_value REAL
            )
        ''')
        
        logger.info("Created message level table: emoji_analysis")
    
    def _create_session_level_tables(self):
        """Create session level analysis tables."""
        
        # PRODUCTION TABLE: Session level metrics with exact 35 columns from session_level_features.py
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_level_metrics (
                unique_session_id TEXT PRIMARY KEY,
                fan_id TEXT,
                chatter_name TEXT,
                model_name TEXT,
                session_start TIMESTAMP,
                session_end TIMESTAMP,
                total_messages INTEGER,
                fan_messages INTEGER,
                chatter_messages INTEGER,
                session_revenue REAL,
                session_tips REAL,
                made_purchase BOOLEAN,
                session_duration_minutes REAL,
                fan_to_chatter_ratio REAL,
                session_value REAL,
                session_length_category TEXT,
                avg_response_time REAL,
                median_response_time REAL,
                min_response_time REAL,
                max_response_time REAL,
                response_count INTEGER,
                avg_fan_msg_length REAL,
                total_fan_msg_length REAL,
                max_fan_msg_length REAL,
                avg_chatter_msg_length REAL,
                total_chatter_msg_length REAL,
                max_chatter_msg_length REAL,
                total_fan_emojis INTEGER,
                avg_fan_emojis REAL,
                total_chatter_emojis INTEGER,
                avg_chatter_emojis REAL,
                fan_media_count INTEGER,
                chatter_media_count INTEGER,
                engagement_intensity REAL,
                engagement_category TEXT
            )
        ''')
        
        logger.info("Created session level table: session_level_metrics (35 columns)")
    
    
    def _create_metadata_table(self):
        """Create metadata table for tracking analysis runs."""
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_type TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                records_processed INTEGER,
                status TEXT,
                error_message TEXT,
                data_file TEXT,
                parameters TEXT
            )
        ''')
        
        # Create indexes for better query performance
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_enhanced_fan_id ON enhanced_fan_analysis(fan_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_enhanced_fan_tier ON enhanced_fan_analysis(tier)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_enhanced_fan_activity ON enhanced_fan_analysis(activity_status)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_fan ON session_level_metrics(fan_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_model ON session_level_metrics(model_name)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_emoji_fan ON emoji_analysis(fan_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_emoji_datetime ON emoji_analysis(datetime)')
        
        logger.info("Created metadata table and indexes")
    
    def drop_all_tables(self):
        """Drop all tables (use with caution!)."""
        self.connect()
        
        # Get all table names
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = self.cursor.fetchall()
        
        # Drop each table
        for table in tables:
            self.cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")
            logger.info(f"Dropped table: {table[0]}")
        
        self.conn.commit()
        self.close()
        logger.info("All tables dropped")
    
    def get_table_info(self, table_name: str):
        """Get information about a specific table."""
        self.connect()
        
        # Get column info
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = self.cursor.fetchall()
        
        # Get row count
        self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = self.cursor.fetchone()[0]
        
        self.close()
        
        return {
            'columns': columns,
            'row_count': row_count
        }
    
    def list_all_tables(self):
        """List all tables in the database."""
        self.connect()
        
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [table[0] for table in self.cursor.fetchall()]
        
        self.close()
        
        return tables


if __name__ == "__main__":
    # Create database schema when run directly
    schema = DatabaseSchema()
    
    # Drop existing tables if needed (uncomment with caution!)
    # schema.drop_all_tables()
    
    # Create all tables
    schema.create_all_tables()
    
    # List all created tables
    tables = schema.list_all_tables()
    print(f"\n✅ Created {len(tables)} tables:")
    for table in tables:
        print(f"  - {table}")