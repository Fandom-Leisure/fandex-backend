#!/usr/bin/env python3
"""
DATABASE SCHEMA DEFINITIONS FOR FANDOM ANALYTICS
=================================================
SQLite table definitions that match the exact outputs from analysis scripts.
All tables preserve the original data structure and column names.
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
        self._create_scripts_tables()
        self._create_metadata_table()
        
        self.conn.commit()
        self.close()
        logger.info("All tables created successfully")
    
    def _create_fan_level_tables(self):
        """Create fan level analysis tables - PRODUCTION VERSION."""
        
        # PRODUCTION TABLE: Enhanced fan analysis (comprehensive metrics)
        # This is the ONLY fan-level table used in production
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_fan_analysis (
                fan_id TEXT PRIMARY KEY,
                total_revenue REAL,
                total_tips REAL,
                total_spending REAL,
                tier TEXT,
                first_interaction TIMESTAMP,
                last_interaction TIMESTAMP,
                total_interactions INTEGER,
                fan_messages_sent INTEGER,
                chatter_messages_received INTEGER,
                days_active INTEGER,
                avg_daily_value REAL,
                messages_per_day REAL,
                days_since_last_interaction INTEGER,
                activity_status TEXT,
                churn_risk TEXT,
                peak_hour INTEGER,
                most_active_day TEXT,
                weekend_activity_pct REAL,
                night_owl_score REAL,
                avg_response_time_minutes REAL,
                conversation_starter_pct REAL,
                emoji_usage_rate REAL,
                avg_message_length REAL,
                questions_asked INTEGER,
                exclamations_used INTEGER,
                media_sent_count INTEGER,
                lifetime_value_score REAL,
                engagement_score REAL,
                combined_score REAL,
                priority_segment TEXT,
                outreach_recommendation TEXT,
                estimated_next_purchase_days INTEGER,
                revenue_potential TEXT
            )
        ''')
        
        # The following tables are commented out for production
        # They were used in development but are now consolidated into enhanced_fan_analysis
        
        # # Main fan analysis summary table - NOT USED IN PRODUCTION
        # self.cursor.execute('''
        #     CREATE TABLE IF NOT EXISTS fan_analysis_summary (
        #         fan_id TEXT PRIMARY KEY,
        #         total_revenue REAL,
        #         total_tips REAL,
        #         total_spending REAL,
        #         tier TEXT,
        #         first_interaction TIMESTAMP,
        #         last_interaction TIMESTAMP,
        #         total_interactions INTEGER,
        #         fan_messages_sent INTEGER,
        #         chatter_messages_received INTEGER,
        #         days_active INTEGER,
        #         avg_daily_value REAL,
        #         messages_per_day REAL
        #     )
        # ''')
        
        # # Detailed fan analysis table - NOT USED IN PRODUCTION
        # self.cursor.execute('''
        #     CREATE TABLE IF NOT EXISTS fan_analysis_detailed (
        #         fan_id TEXT PRIMARY KEY,
        #         total_spending REAL,
        #         tier TEXT,
        #         total_interactions INTEGER,
        #         days_active INTEGER,
        #         avg_daily_value REAL,
        #         messages_per_day REAL,
        #         peak_hour INTEGER,
        #         most_active_day TEXT,
        #         weekend_activity_pct REAL,
        #         night_owl_score REAL,
        #         avg_response_time_minutes REAL,
        #         conversation_starter_pct REAL,
        #         emoji_usage_rate REAL,
        #         avg_message_length REAL,
        #         questions_asked INTEGER,
        #         exclamations_used INTEGER,
        #         media_sent_count INTEGER
        #     )
        # ''')
        
        # # Fan segments table - NOT USED IN PRODUCTION (segments are in enhanced_fan_analysis)
        # self.cursor.execute('''
        #     CREATE TABLE IF NOT EXISTS fan_segments (
        #         fan_id TEXT,
        #         segment_type TEXT,
        #         total_spending REAL,
        #         tier TEXT,
        #         days_since_last_interaction INTEGER,
        #         activity_status TEXT,
        #         churn_risk TEXT,
        #         lifetime_value_score REAL,
        #         engagement_score REAL,
        #         combined_score REAL,
        #         priority_segment TEXT,
        #         outreach_recommendation TEXT,
        #         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        #         PRIMARY KEY (fan_id, segment_type)
        #     )
        # ''')
        
        logger.info("Created fan level table: enhanced_fan_analysis")
    
    def _create_message_level_tables(self):
        """Create message level analysis tables."""
        
        # Emoji analysis table (per message)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS emoji_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fan_id TEXT,
                model_name TEXT,
                chatter_name TEXT,
                datetime TIMESTAMP,
                fan_message TEXT,
                chatter_message TEXT,
                fan_emojis TEXT,
                chatter_emojis TEXT,
                fan_emoji_count INTEGER,
                chatter_emoji_count INTEGER,
                fan_emoji_str TEXT,
                chatter_emoji_str TEXT,
                fan_has_emoji BOOLEAN,
                chatter_has_emoji BOOLEAN
            )
        ''')
        
        # Emoji analysis summary
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS emoji_analysis_summary (
                metric TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # Message level with media analysis
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS message_level_with_media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fan_id TEXT,
                model_name TEXT,
                chatter_name TEXT,
                datetime TIMESTAMP,
                fan_message TEXT,
                chatter_message TEXT,
                revenue REAL,
                tips REAL,
                fan_has_emoji BOOLEAN,
                chatter_has_emoji BOOLEAN,
                fan_emoji_count INTEGER,
                chatter_emoji_count INTEGER,
                has_media_trigger BOOLEAN,
                media_trigger_type TEXT,
                responded_with_purchase BOOLEAN,
                time_to_purchase_minutes REAL,
                purchase_amount REAL
            )
        ''')
        
        # Media analysis summary
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS media_analysis_summary (
                metric TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # Media by chatter
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS media_by_chatter (
                chatter_name TEXT PRIMARY KEY,
                total_triggers INTEGER,
                conversions INTEGER,
                conversion_rate REAL,
                avg_purchase_amount REAL,
                total_revenue REAL,
                avg_time_to_purchase REAL
            )
        ''')
        
        # Media by model
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS media_by_model (
                model_name TEXT PRIMARY KEY,
                total_triggers INTEGER,
                conversions INTEGER,
                conversion_rate REAL,
                avg_purchase_amount REAL,
                total_revenue REAL,
                avg_time_to_purchase REAL
            )
        ''')
        
        # Media hourly distribution
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS media_hourly_distribution (
                hour INTEGER PRIMARY KEY,
                trigger_count INTEGER,
                conversion_count INTEGER,
                conversion_rate REAL
            )
        ''')
        
        # Fan media behavior
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS fan_media_behavior (
                fan_id TEXT PRIMARY KEY,
                total_triggers_received INTEGER,
                total_conversions INTEGER,
                conversion_rate REAL,
                avg_purchase_amount REAL,
                total_spent_on_media REAL,
                avg_response_time REAL,
                preferred_media_type TEXT
            )
        ''')
        
        # Media triggers analysis tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS fan_media_triggers (
                fan_id TEXT PRIMARY KEY,
                triggers_received INTEGER,
                purchases_made INTEGER,
                conversion_rate REAL,
                total_spent REAL,
                avg_purchase REAL,
                avg_response_time_min REAL
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS chatter_media_triggers (
                chatter_name TEXT PRIMARY KEY,
                triggers_sent INTEGER,
                conversions INTEGER,
                conversion_rate REAL,
                revenue_generated REAL,
                avg_purchase REAL
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS all_media_triggers_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_type TEXT,
                category TEXT,
                value REAL,
                count INTEGER,
                percentage REAL
            )
        ''')
        
        logger.info("Created message level tables")
    
    def _create_session_level_tables(self):
        """Create session level analysis tables."""
        
        # Session level metrics
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_level_metrics (
                unique_session_id TEXT PRIMARY KEY,
                fan_id TEXT,
                model_name TEXT,
                chatter_name TEXT,
                session_start TIMESTAMP,
                session_end TIMESTAMP,
                message_count INTEGER,
                fan_messages INTEGER,
                chatter_messages INTEGER,
                avg_fan_msg_length REAL,
                max_fan_msg_length REAL,
                avg_chatter_msg_length REAL,
                max_chatter_msg_length REAL,
                total_revenue REAL,
                revenue_only REAL,
                tips_only REAL,
                generated_revenue BOOLEAN,
                duration_minutes REAL,
                messages_per_minute REAL,
                response_rate REAL,
                avg_response_time_minutes REAL
            )
        ''')
        
        # Fan level session metrics (aggregated)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS fan_level_session_metrics (
                fan_id TEXT PRIMARY KEY,
                total_sessions INTEGER,
                total_messages INTEGER,
                total_revenue REAL,
                avg_session_duration REAL,
                avg_messages_per_session REAL,
                avg_revenue_per_session REAL,
                conversion_rate REAL,
                total_fan_messages INTEGER,
                total_chatter_messages INTEGER,
                avg_response_time REAL,
                preferred_hour INTEGER,
                preferred_day TEXT
            )
        ''')
        
        # Session analysis summary
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_analysis_summary (
                metric TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # Session analysis complete (JSON storage)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_analysis_complete (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_type TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Hourly session patterns
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS hourly_session_patterns (
                hour INTEGER PRIMARY KEY,
                session_count INTEGER,
                avg_duration REAL,
                avg_messages REAL,
                avg_revenue REAL,
                conversion_rate REAL
            )
        ''')
        
        # Session by model
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_by_model (
                model_name TEXT PRIMARY KEY,
                total_sessions INTEGER,
                total_revenue REAL,
                avg_session_duration REAL,
                avg_messages_per_session REAL,
                avg_revenue_per_session REAL,
                conversion_rate REAL
            )
        ''')
        
        # Session category distribution
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_category_distribution (
                category TEXT PRIMARY KEY,
                session_count INTEGER,
                percentage REAL,
                avg_duration REAL,
                avg_revenue REAL
            )
        ''')
        
        logger.info("Created session level tables")
    
    def _create_scripts_tables(self):
        """Create scripts analysis tables."""
        
        # Script statistics
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS script_statistics (
                unique_script_id TEXT PRIMARY KEY,
                fan_id TEXT,
                chatter_name TEXT,
                model_name TEXT,
                script_start TIMESTAMP,
                script_end TIMESTAMP,
                total_messages INTEGER,
                total_revenue REAL,
                revenue_only REAL,
                tips_only REAL,
                generated_revenue BOOLEAN,
                fan_messages INTEGER,
                chatter_messages INTEGER,
                duration_minutes REAL,
                messages_per_minute REAL,
                revenue_per_message REAL,
                revenue_efficiency REAL
            )
        ''')
        
        # Model summary
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_summary (
                model_name TEXT PRIMARY KEY,
                total_scripts INTEGER,
                revenue_generating_scripts INTEGER,
                total_revenue REAL,
                avg_revenue_per_script REAL,
                conversion_rate REAL,
                avg_messages_per_script REAL,
                avg_duration_minutes REAL
            )
        ''')
        
        # Top scripts per model (JSON storage)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_top_scripts (
                model_name TEXT,
                script_rank INTEGER,
                script_id TEXT,
                revenue REAL,
                messages TEXT,
                fan_messages TEXT,
                chatter_messages TEXT,
                duration_minutes REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model_name, script_rank)
            )
        ''')
        
        # All models top 5 scripts (JSON storage)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS all_models_top_scripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Analysis summary (JSON storage)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS scripts_analysis_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_type TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        logger.info("Created scripts analysis tables")
    
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
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_fan_id ON fan_analysis_summary(fan_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_fan_tier ON fan_analysis_summary(tier)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_fan ON session_level_metrics(fan_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_session_model ON session_level_metrics(model_name)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_script_model ON script_statistics(model_name)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_fan ON message_level_with_media(fan_id)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_datetime ON message_level_with_media(datetime)')
        
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