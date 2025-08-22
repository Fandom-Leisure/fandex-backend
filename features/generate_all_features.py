#!/usr/bin/env python3
"""
GENERATE ALL FEATURES - Main Orchestrator Script
================================================
Runs all feature generation scripts and creates the complete SQLite database.
This is the main script that data analysts should run.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import sqlite3
import pandas as pd

# Import all feature generators
from database_schema import DatabaseSchema
from fan_level_features import FanLevelFeatures
from message_level_features import MessageLevelFeatures
from session_level_features import SessionLevelFeatures


class FeatureOrchestrator:
    """Orchestrates the generation of all features."""
    
    def __init__(self, data_path: str = None, db_path: str = None, 
                 reset_db: bool = False):
        # Set default paths - using message_level file for all features
        self.data_path = data_path or "../data/raw/all_chatlogs_message_level.pkl"
        self.db_path = db_path or "../data/processed/features.db"
        self.reset_db = reset_db
        
        # Convert to absolute paths
        self.data_path = Path(self.data_path).resolve()
        self.db_path = Path(self.db_path).resolve()
        
        # Create directories if they don't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Track timing
        self.start_time = None
        self.feature_times = {}
    
    def validate_data_file(self):
        """Check if data file exists."""
        if not self.data_path.exists():
            print(f"❌ Data file not found: {self.data_path}")
            print("\nPlease ensure your data file is in one of these locations:")
            print("  1. ../data/raw/all_chatlogs_message_level.pkl (default)")
            print("  2. Specify custom path with --data argument")
            return False
        
        print(f"✅ Data file found: {self.data_path}")
        print(f"📊 File size: {self.data_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    
    def setup_database(self):
        """Initialize database schema."""
        print("\n" + "="*60)
        print("🗄️  SETTING UP DATABASE")
        print("="*60)
        
        schema = DatabaseSchema(str(self.db_path))
        
        if self.reset_db and self.db_path.exists():
            print("⚠️  Resetting database (removing existing tables)...")
            schema.drop_all_tables()
        
        print("📝 Creating database schema...")
        schema.create_all_tables()
        
        # List created tables
        tables = schema.list_all_tables()
        print(f"✅ Database ready with {len(tables)} tables")
    
    def generate_fan_features(self):
        """Generate fan-level features."""
        print("\n" + "="*60)
        print("👥 GENERATING FAN LEVEL FEATURES")
        print("="*60)
        
        start = datetime.now()
        
        try:
            generator = FanLevelFeatures(
                data_path=str(self.data_path),
                db_path=str(self.db_path)
            )
            generator.generate_features()
            
            self.feature_times['fan_level'] = (datetime.now() - start).total_seconds()
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate fan features: {e}")
            self.feature_times['fan_level'] = None
            return False
    
    def generate_message_features(self):
        """Generate message-level features."""
        print("\n" + "="*60)
        print("💬 GENERATING MESSAGE LEVEL FEATURES")
        print("="*60)
        
        start = datetime.now()
        
        try:
            generator = MessageLevelFeatures(
                data_path=str(self.data_path),
                db_path=str(self.db_path)
            )
            generator.generate_features()
            
            self.feature_times['message_level'] = (datetime.now() - start).total_seconds()
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate message features: {e}")
            self.feature_times['message_level'] = None
            return False
    
    def generate_session_features(self):
        """Generate session-level features."""
        print("\n" + "="*60)
        print("🔄 GENERATING SESSION LEVEL FEATURES")
        print("="*60)
        
        start = datetime.now()
        
        try:
            generator = SessionLevelFeatures(
                data_path=str(self.data_path),
                db_path=str(self.db_path)
            )
            generator.generate_features()
            
            self.feature_times['session_level'] = (datetime.now() - start).total_seconds()
            return True
            
        except Exception as e:
            print(f"❌ Failed to generate session features: {e}")
            self.feature_times['session_level'] = None
            return False
    
    def verify_database(self):
        """Verify database contents."""
        print("\n" + "="*60)
        print("🔍 VERIFYING DATABASE")
        print("="*60)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        print(f"\n📊 Database contains {len(tables)} tables:")
        
        total_records = 0
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            total_records += count
            print(f"  • {table_name:40} : {count:10,} records")
        
        print(f"\n📈 Total records across all tables: {total_records:,}")
        
        # Get database file size
        db_size = self.db_path.stat().st_size / (1024*1024)
        print(f"💾 Database file size: {db_size:.2f} MB")
        
        conn.close()
    
    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*60)
        print("📋 EXECUTION SUMMARY")
        print("="*60)
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        print("\n⏱️  Feature Generation Times:")
        for feature, time_taken in self.feature_times.items():
            if time_taken is not None:
                print(f"  • {feature:20} : {time_taken:8.2f} seconds")
            else:
                print(f"  • {feature:20} : ❌ Failed")
        
        print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
        print(f"📁 Database location: {self.db_path}")
        
        # Check if all features were generated successfully
        success_count = sum(1 for t in self.feature_times.values() if t is not None)
        total_count = len(self.feature_times)
        
        if success_count == total_count:
            print("\n✅ ALL FEATURES GENERATED SUCCESSFULLY!")
        else:
            print(f"\n⚠️  {success_count}/{total_count} features generated successfully")
    
    def run(self, features_to_run=None):
        """Run the complete feature generation pipeline."""
        self.start_time = datetime.now()
        
        print("\n" + "="*70)
        print("🚀 FANDOM ANALYTICS - FEATURE GENERATION PIPELINE")
        print("="*70)
        print(f"📅 Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Data source: {self.data_path}")
        print(f"🗄️  Database: {self.db_path}")
        
        # Validate data file
        if not self.validate_data_file():
            return False
        
        # Setup database
        self.setup_database()
        
        # Determine which features to generate
        all_features = ['fan', 'message', 'session']
        
        if features_to_run is None or 'all' in features_to_run:
            features_to_run = all_features
        
        # Generate features
        print(f"\n📝 Generating features: {', '.join(features_to_run)}")
        
        if 'fan' in features_to_run:
            self.generate_fan_features()
        
        if 'message' in features_to_run:
            self.generate_message_features()
        
        if 'session' in features_to_run:
            self.generate_session_features()
        
        # Verify database
        self.verify_database()
        
        # Print summary
        self.print_summary()
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate all features for Fandom Analytics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_all_features.py                    # Generate all features
  python generate_all_features.py --reset            # Reset DB and regenerate
  python generate_all_features.py --features fan     # Generate only fan features
  python generate_all_features.py --features fan message  # Generate fan and message features
  python generate_all_features.py --data /path/to/data.pkl  # Use custom data file
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        help='Path to the input data file (pickle format)',
        default='../data/raw/all_chatlogs_message_level.pkl'
    )
    
    parser.add_argument(
        '--database', '--db',
        help='Path to the output SQLite database',
        default='../data/processed/features.db'
    )
    
    parser.add_argument(
        '--reset', '-r',
        action='store_true',
        help='Reset database (drop all tables) before generating features'
    )
    
    parser.add_argument(
        '--features', '-f',
        nargs='+',
        choices=['all', 'fan', 'message', 'session'],
        default=['all'],
        help='Which features to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = FeatureOrchestrator(
        data_path=args.data,
        db_path=args.database,
        reset_db=args.reset
    )
    
    # Run feature generation
    success = orchestrator.run(features_to_run=args.features)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()