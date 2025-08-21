from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, TypedDict
from collections import Counter
import re
import sqlite3
import json
import pytz
from pathlib import Path
import traceback
from .models import (
    FanSummary, MoodType, PurchaseReadiness, 
    Message, FanHistory, DashboardStats,
    SpendingFrequency, EngagementBehavior, 
    EmotionalAttachment, LifecycleStage,
    LTVSegment, FanSegment, SegmentationConfig
)
from .data_processor import DataProcessor

class Purchase(TypedDict):
    """Standardized purchase record format."""
    datetime: datetime
    price: float
    tips: float
    total: float
    message: Optional[str]

class FanAnalyzer:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.db_path = Path("dashboard/fan_notes.db")
        self._init_notes_db()
        self._timezone = None  # Cache for timezone
    
    def _init_notes_db(self):
        """Initialize SQLite database for personal notes and segmentation."""
        self.db_path.parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fan notes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fan_notes (
                fan_id TEXT PRIMARY KEY,
                notes TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Segmentation config table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS segmentation_config (
                creator_id TEXT,
                version INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config TEXT,
                timezone TEXT DEFAULT 'Asia/Manila',
                PRIMARY KEY (creator_id, version)
            )
        ''')
        
        # Fan segment history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fan_segment_history (
                fan_id TEXT,
                model_name TEXT,
                segment_code TEXT,
                spending_frequency TEXT,
                engagement_behavior TEXT,
                emotional_attachment TEXT,
                lifecycle_stage TEXT,
                ltv_segment TEXT,
                valid_from TIMESTAMP,
                valid_to TIMESTAMP,
                version INTEGER,
                reason TEXT,
                features TEXT,
                is_provisional BOOLEAN,
                PRIMARY KEY (fan_id, model_name, valid_from)
            )
        ''')
        
        # Add ltv_segment column if it doesn't exist (migration)
        cursor.execute("PRAGMA table_info(fan_segment_history)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'ltv_segment' not in columns:
            cursor.execute('ALTER TABLE fan_segment_history ADD COLUMN ltv_segment TEXT DEFAULT "M"')
            conn.commit()
        
        # Insert default config if none exists
        cursor.execute('''
            INSERT OR IGNORE INTO segmentation_config 
            (creator_id, version, config, timezone) 
            VALUES ('default', 1, ?, 'Asia/Manila')
        ''', (json.dumps(SegmentationConfig().model_dump()),))
        
        conn.commit()
        conn.close()
    
    def _get_timezone(self) -> pytz.timezone:
        """Get creator timezone from config (cached)."""
        if self._timezone is None:
            config = self._get_segmentation_config()
            self._timezone = pytz.timezone(config.timezone)
        return self._timezone
    
    def _get_timezone_aware_now(self) -> datetime:
        """Get latest datetime from the data in creator's timezone."""
        tz = self._get_timezone()
        
        # Get the latest date from the messages_with_emotions.pkl data
        try:
            # Access the data through data_processor
            latest_date = self.data_processor.emotions_df['datetime'].max()
            
            # Convert pandas Timestamp to Python datetime
            if hasattr(latest_date, 'to_pydatetime'):
                latest_date = latest_date.to_pydatetime()
            
            # Handle timezone - the data might be timezone-naive
            if latest_date.tzinfo is None:
                latest_date = tz.localize(latest_date)
            else:
                latest_date = latest_date.astimezone(tz)
                
            return latest_date
        except Exception as e:
            # Fallback to current time if there's any issue
            import logging
            logging.warning(f"Failed to get latest date from data: {e}. Using current time.")
            return datetime.now(tz)
    
    def _to_tz(self, dt: datetime) -> datetime:
        tz = self._get_timezone()
        return tz.localize(dt) if dt.tzinfo is None else dt.astimezone(tz)

    def _ensure_purchase_ordering(self, purchases: List[Dict]) -> List[Dict]:
        """Ensure purchases are sorted newest-first."""
        return sorted(purchases, key=lambda p: p['datetime'], reverse=True)
    
    def _convert_message_to_purchase(self, msg: Message) -> Purchase:
        """Convert a Message object to a standardized Purchase dict."""
        return Purchase(
            datetime=msg.datetime,
            price=msg.price,
            tips=msg.tips,
            total=msg.price + msg.tips,
            message=msg.message
        )
    
    def analyze_fan(self, fan_id: str, model_name: Optional[str] = None) -> FanSummary:
        """Analyze a fan and return comprehensive summary with all 8 features."""
        messages = self.data_processor.get_fan_messages(fan_id, model_name)
        if not messages:
            return None
        
        purchase_history = self._ensure_purchase_ordering(
            self.data_processor.get_fan_purchase_history(fan_id, model_name)
        )
        conv_stats = self.data_processor.get_fan_conversation_stats(fan_id, model_name)
        
        total_spent = sum(p['total'] for p in purchase_history)
        
        days_since_purchase = self._calculate_days_since_purchase(purchase_history)
        
        last_msg = messages[-1] if messages else None
        last_message = last_msg.message if last_msg else None
        last_message_time = last_msg.datetime if last_msg else None
        
        response_pattern = self._analyze_response_pattern(messages)
        conversation_streak = self._calculate_conversation_streak(messages)
        
        current_mood, mood_emoji = self._analyze_current_mood(messages[-10:] if len(messages) > 10 else messages)
        
        purchase_readiness, readiness_score = self._calculate_purchase_readiness(
            messages, purchase_history, days_since_purchase
        )
        
        best_time = self._find_best_chat_time(messages)
        successful_topics = self._extract_successful_topics(messages, purchase_history)
        message_style = self._analyze_message_style(messages)
        
        next_action = self._suggest_next_action(
            current_mood, purchase_readiness, days_since_purchase, conversation_streak
        )
        
        recent_topics = self._extract_recent_topics(messages[-20:] if len(messages) > 20 else messages)
        
        personal_notes = self._get_personal_notes(fan_id)
        
        # Get segmentation
        segmentation = self._compute_fan_segmentation(
            fan_id, model_name or "", messages, purchase_history, conv_stats
        )
        
        return FanSummary(
            fan_id=fan_id,
            model_name=model_name or (messages[0].sender_id if messages else ""),
            total_spent=total_spent,
            days_since_last_purchase=days_since_purchase,
            last_message=last_message,
            last_message_time=last_message_time,
            response_pattern=response_pattern,
            conversation_streak=conversation_streak,
            current_mood=current_mood,
            mood_emoji=mood_emoji,
            purchase_readiness=purchase_readiness,
            purchase_readiness_score=readiness_score,
            best_time_to_chat=best_time,
            successful_topics=successful_topics,
            preferred_message_style=message_style,
            next_action_suggestion=next_action,
            recent_topics=recent_topics,
            personal_notes=personal_notes,
            message_count=len(messages),
            avg_response_time_minutes=conv_stats.get('avg_response_time_minutes'),
            total_conversations=conv_stats.get('total_conversations', 0),
            segmentation=segmentation
        )
    
    def _calculate_days_since_purchase(self, purchase_history: List[Dict]) -> Optional[int]:
        """Calculate days since last purchase. Assumes purchase_history is sorted newest-first."""
        if not purchase_history:
            return None
        last_purchase = purchase_history[0]['datetime']
        tz = self._get_timezone()
        # Handle timezone-naive datetimes
        if last_purchase.tzinfo is None:
            last_purchase = tz.localize(last_purchase)
        current_time = self._get_timezone_aware_now()
        return (current_time - last_purchase).days
    
    def _analyze_response_pattern(self, messages: List[Message]) -> str:
        """Analyze fan's response pattern."""
        if len(messages) < 5:
            return "New fan"
        
        response_times = []
        for i in range(1, len(messages)):
            if messages[i].sender_type.value == 'fan' and messages[i-1].sender_type.value != 'fan':
                time_diff = (messages[i].datetime - messages[i-1].datetime).total_seconds() / 60
                if time_diff < 180:
                    response_times.append(time_diff)
        
        if not response_times:
            return "Passive"
        
        avg_response = sum(response_times) / len(response_times)
        if avg_response < 5:
            return "Very responsive"
        elif avg_response < 15:
            return "Responsive"
        elif avg_response < 60:
            return "Moderate"
        else:
            return "Slow responder"
    
    def _calculate_conversation_streak(self, messages: List[Message]) -> int:
        """Calculate current conversation streak in days."""
        if not messages:
            return 0
        
        tz = self._get_timezone()
        # Convert message datetimes to timezone-aware dates
        dates = []
        for msg in messages:
            msg_dt = msg.datetime
            if msg_dt.tzinfo is None:
                msg_dt = tz.localize(msg_dt)
            else:
                msg_dt = msg_dt.astimezone(tz)
            dates.append(msg_dt.date())
        
        dates = sorted(set(dates), reverse=True)
        streak = 0
        current_date = self._get_timezone_aware_now().date()
        
        for date in dates:
            if (current_date - date).days == streak:
                streak += 1
            else:
                break
        
        return streak
    
    def _analyze_current_mood(self, recent_messages: List[Message]) -> Tuple[MoodType, str]:
        """Analyze current mood based on recent messages and emotions."""
        if not recent_messages:
            return MoodType.NEUTRAL, "😐"
        
        fan_messages = [msg for msg in recent_messages if msg.sender_type.value == 'fan']
        if not fan_messages:
            return MoodType.NEUTRAL, "😐"
        
        emotion_scores = {
            'joy': 0, 'love': 0, 'optimism': 0,
            'sadness': 0, 'anger': 0, 'fear': 0,
            'surprise': 0, 'anticipation': 0
        }
        
        for msg in fan_messages[-5:]:
            if msg.emotions:
                for emotion, score in msg.emotions.items():
                    if emotion in emotion_scores:
                        emotion_scores[emotion] += score
        
        positive_score = emotion_scores['joy'] + emotion_scores['love'] + emotion_scores['optimism']
        negative_score = emotion_scores['sadness'] + emotion_scores['anger'] + emotion_scores['fear']
        excitement_score = emotion_scores['surprise'] + emotion_scores['anticipation']
        
        if positive_score > negative_score * 2:
            if emotion_scores['love'] > emotion_scores['joy']:
                return MoodType.LOVING, "🥰"
            return MoodType.HAPPY, "😊"
        elif negative_score > positive_score * 2:
            if emotion_scores['anger'] > emotion_scores['sadness']:
                return MoodType.ANGRY, "😠"
            return MoodType.SAD, "😔"
        elif excitement_score > positive_score:
            return MoodType.EXCITED, "🔥"
        else:
            return MoodType.NEUTRAL, "😐"
    
    def _calculate_purchase_readiness(self, messages: List[Message], 
                                    purchase_history: List[Dict],
                                    days_since_purchase: Optional[int]) -> Tuple[PurchaseReadiness, int]:
        """Calculate purchase readiness score and category."""
        score = 5
        
        recent_messages = messages[-20:] if len(messages) > 20 else messages
        engagement_rate = len([m for m in recent_messages if m.sender_type.value == 'fan']) / len(recent_messages) if recent_messages else 0
        
        if engagement_rate > 0.4:
            score += 2
        elif engagement_rate > 0.2:
            score += 1
        
        mood, _ = self._analyze_current_mood(recent_messages)
        if mood in [MoodType.HAPPY, MoodType.EXCITED, MoodType.LOVING]:
            score += 2
        elif mood == MoodType.NEUTRAL:
            score += 1
        elif mood == MoodType.ANGRY:
            score -= 2
        
        # Distinguish between never purchased vs no recent purchase
        if days_since_purchase is not None:
            if days_since_purchase > 30:
                score += 2
            elif days_since_purchase > 14:
                score += 1
            elif days_since_purchase < 3:
                score -= 4
        else:
            # Check if this is truly a never-buyer or just no data
            has_any_purchases = len(purchase_history) > 0
            if not has_any_purchases:
                # Never purchased - check account age
                if messages:
                    first_msg_date = messages[0].datetime
                    tz = self._get_timezone()
                    if first_msg_date.tzinfo is None:
                        first_msg_date = tz.localize(first_msg_date)
                    account_age = (self._get_timezone_aware_now() - first_msg_date).days
                    
                    if account_age < 7:
                        score += 1  # New fan, still warming up
                    elif account_age < 30:
                        score += 0  # Been around but no purchases yet
                    else:
                        score -= 1  # Long-time non-buyer
                else:
                    score += 0  # No data to determine
        
        if len(messages) > 50:
            score += 1
        
        if score >= 8:
            return PurchaseReadiness.HIGH, score
        elif score >= 5:
            return PurchaseReadiness.MEDIUM, score
        else:
            return PurchaseReadiness.LOW, score
    
    def _find_best_chat_time(self, messages: List[Message]) -> str:
        """Find the best time to chat based on fan messages in creator timezone."""
        if not messages:
            return "No data yet"
        
        tz = self._get_timezone()
        fan_messages = [m for m in messages if m.sender_type.value == 'fan']
        if not fan_messages:
            return "No fan messages yet"
        
        hours = []
        for msg in fan_messages:
            dt = msg.datetime
            if dt.tzinfo is None:
                dt = tz.localize(dt)
            else:
                dt = dt.astimezone(tz)
            hours.append(dt.hour)
        
        if not hours:
            return "Anytime"
        
        peak_hour = Counter(hours).most_common(1)[0][0]
        
        if 5 <= peak_hour < 12:
            return f"Morning ({peak_hour}:00-{peak_hour+1}:00)"
        elif 12 <= peak_hour < 17:
            return f"Afternoon ({peak_hour}:00-{peak_hour+1}:00)"
        elif 17 <= peak_hour < 22:
            return f"Evening ({peak_hour}:00-{peak_hour+1}:00)"
        else:
            return f"Night ({peak_hour}:00-{peak_hour+1}:00)"
    
    def _extract_successful_topics(self, messages: List[Message], 
                                  purchase_history: List[Dict]) -> List[str]:
        """Extract topics that led to purchases."""
        if not purchase_history:
            return ["Build rapport first"]
        
        successful_topics = []
        for purchase in purchase_history[:5]:
            purchase_time = purchase['datetime']
            
            relevant_messages = [
                msg for msg in messages 
                if purchase_time - timedelta(hours=2) <= msg.datetime <= purchase_time
            ]
            
            topics = []
            for msg in relevant_messages:
                if msg.message:
                    words = re.findall(r'\b\w{4,}\b', msg.message.lower())
                    topics.extend(words)
            
            if topics:
                topic_counts = Counter(topics)
                common_topics = [topic for topic, _ in topic_counts.most_common(3)]
                successful_topics.extend(common_topics)
        
        if not successful_topics:
            return ["Compliments", "Personal interest", "Exclusive content"]
        
        topic_summary = Counter(successful_topics)
        return [topic for topic, _ in topic_summary.most_common(3)]
    
    def _analyze_message_style(self, messages: List[Message]) -> str:
        """Analyze preferred message style."""
        fan_messages = [msg for msg in messages if msg.sender_type.value == 'fan' and msg.message]
        if not fan_messages:
            return "Unknown"
        
        total_length = sum(len(msg.message) for msg in fan_messages)
        avg_length = total_length / len(fan_messages)
        
        emoji_count = sum(1 for msg in fan_messages if any(char in msg.message for char in '😊🥰❤️💕🔥😍'))
        emoji_rate = emoji_count / len(fan_messages)
        
        if avg_length < 20:
            style = "Short & sweet"
        elif avg_length < 50:
            style = "Conversational"
        else:
            style = "Detailed"
        
        if emoji_rate > 0.5:
            style += " with emojis"
        
        return style
    
    def _suggest_next_action(self, mood: MoodType, readiness: PurchaseReadiness,
                           days_since_purchase: Optional[int], streak: int) -> str:
        """Suggest the next action based on current state."""
        if readiness == PurchaseReadiness.HIGH:
            if mood in [MoodType.HAPPY, MoodType.EXCITED, MoodType.LOVING]:
                return "🎯 Perfect time for exclusive content offer!"
            else:
                return "💬 Boost mood first, then offer content"
        
        elif readiness == PurchaseReadiness.MEDIUM:
            if streak > 3:
                return "🔥 Reward loyalty with special discount"
            elif mood == MoodType.NEUTRAL:
                return "✨ Share something exciting to engage"
            else:
                return "💭 Continue building connection"
        
        else:
            if days_since_purchase and days_since_purchase < 7:
                return "⏰ Too soon - focus on satisfaction"
            elif streak == 0:
                return "👋 Re-engage with personal message"
            else:
                return "🌱 Nurture relationship first"
    
    def _extract_recent_topics(self, recent_messages: List[Message]) -> List[str]:
        """Extract recent conversation topics."""
        all_words = []
        for msg in recent_messages:
            if msg.message:
                words = re.findall(r'\b\w{4,}\b', msg.message.lower())
                all_words.extend(words)
        
        if not all_words:
            return []
        
        stop_words = {'just', 'like', 'that', 'this', 'what', 'have', 'been', 'your', 'with'}
        filtered_words = [w for w in all_words if w not in stop_words]
        
        word_counts = Counter(filtered_words)
        return [word for word, _ in word_counts.most_common(3)]
    
    def _get_personal_notes(self, fan_id: str) -> str:
        """Get personal notes for a fan."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT notes FROM fan_notes WHERE fan_id = ?", (fan_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else ""
    
    def save_personal_notes(self, fan_id: str, notes: str):
        """Save personal notes for a fan."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO fan_notes (fan_id, notes, updated_at)
            VALUES (?, ?, ?)
        """, (fan_id, notes, self._get_timezone_aware_now()))
        conn.commit()
        conn.close()
    
    def _get_segmentation_config(self, creator_id: str = "default") -> SegmentationConfig:
        """Get current segmentation configuration."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT config FROM segmentation_config 
            WHERE creator_id = ? 
            ORDER BY version DESC 
            LIMIT 1
        """, (creator_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return SegmentationConfig(**json.loads(result[0]))
        return SegmentationConfig()
    
    def _get_rolling_window_data(self, messages: List[Message], days: int = 28) -> Tuple[List[Message], List[Dict]]:
        """Get data for rolling window in creator timezone."""
        config = self._get_segmentation_config()
        tz = pytz.timezone(config.timezone)
        end_date = self._get_timezone_aware_now()
        start_date = end_date - timedelta(days=days)
        
        # Filter messages
        window_messages = []
        for msg in messages:
            try:
                # Handle timezone-naive datetimes
                msg_dt = msg.datetime
                if msg_dt.tzinfo is None:
                    msg_dt = tz.localize(msg_dt)
                else:
                    msg_dt = msg_dt.astimezone(tz)
                
                if msg_dt >= start_date:
                    window_messages.append(msg)
            except Exception:
                # If timezone conversion fails, assume it's in the target timezone
                if msg.datetime >= start_date.replace(tzinfo=None):
                    window_messages.append(msg)
        
        # Convert Message purchases to standardized Purchase dicts
        window_purchases = []
        for msg in window_messages:
            if msg.purchased and msg.price > 0:
                window_purchases.append(self._convert_message_to_purchase(msg))
        
        return window_messages, window_purchases
    
    def _compute_fan_segmentation(self, fan_id: str, model_name: str, 
                                 messages: List[Message], 
                                 purchase_history: List[Dict],
                                 conv_stats: Dict) -> FanSegment:
        """Compute comprehensive fan segmentation."""
        try:
            config = self._get_segmentation_config()
            
            # Get 28-day rolling window data
            window_messages, window_purchases = self._get_rolling_window_data(messages)
            
            # Check if fan is provisional (< 14 days)
            first_msg_date = messages[0].datetime if messages else self._get_timezone_aware_now()
            tz = self._get_timezone()
            if first_msg_date.tzinfo is None:
                first_msg_date = tz.localize(first_msg_date)
            is_provisional = (self._get_timezone_aware_now() - first_msg_date).days < 14
            
            # Analyze each axis
            spending_freq = self._analyze_spending_frequency(window_purchases, purchase_history, config)
            engagement = self._analyze_engagement_behavior(window_messages, window_purchases, conv_stats)
            emotional = self._analyze_emotional_attachment(window_messages, window_purchases, config)
            lifecycle = self._analyze_lifecycle_stage(messages, purchase_history, config)
            
            # Calculate total spent for LTV segment
            total_spent = sum(p.get('price', 0) for p in purchase_history)
            ltv_segment = self._analyze_ltv_segment(total_spent)
            
            # Build segment code
            segment_code = f"{spending_freq.value}-{engagement.value}-{emotional.value}-{lifecycle.value}-{ltv_segment.value}"
            
            # Prepare computation features for audit
            features = {
                "window_days": 28,
                "total_messages": len(window_messages),
                "total_purchases": len(window_purchases),
                "is_provisional": is_provisional,
                "config_version": config.version
            }
            
            segment = FanSegment(
                spending_frequency=spending_freq,
                engagement_behavior=engagement,
                emotional_attachment=emotional,
                lifecycle_stage=lifecycle,
                ltv_segment=ltv_segment,
                segment_code=segment_code,
                is_provisional=is_provisional,
                computation_features=features
            )
            
            # Track segment history
            self._track_segment_change(fan_id, model_name, segment, features)
            
            return segment
        except Exception as e:
            # Log error and return default segment
            import logging
            logging.error(f"Error computing segmentation for {fan_id}: {e}")
            logging.info(traceback.format_exc())
            # Return a default segment on error
            return FanSegment(
                spending_frequency=SpendingFrequency.NW,
                engagement_behavior=EngagementBehavior.L,
                emotional_attachment=EmotionalAttachment.F,
                lifecycle_stage=LifecycleStage.N,
                ltv_segment=LTVSegment.M,
                segment_code="NW-L-F-N-M",
                is_provisional=True,
                computation_features={"error": str(e)}
            )
    
    def _analyze_spending_frequency(self, window_purchases: List[Dict], 
                                  all_purchases: List[Dict],
                                  config: SegmentationConfig) -> SpendingFrequency:
        """Analyze spending frequency based on purchase-days per week."""
        if not window_purchases:
            return SpendingFrequency.TW
        
        # Calculate total spent in window
        window_total = sum(p['price'] + p['tips'] for p in window_purchases)
        if window_total < config.min_profitable_threshold:
            return SpendingFrequency.NW
        
        # Check for In & Out pattern (burst + inactivity)
        if all_purchases and len(all_purchases) > 0:
            # all_purchases is sorted newest-first, so oldest is at the end
            oldest_purchase = all_purchases[-1]['datetime']
            tz = self._get_timezone()
            if oldest_purchase.tzinfo is None:
                oldest_purchase = tz.localize(oldest_purchase)
            days_since_first = (self._get_timezone_aware_now() - oldest_purchase).days
            
            # Check if all purchases were in first 48 hours
            burst_cutoff = oldest_purchase + timedelta(hours=config.burst_window_hours)
            burst_purchases = []
            for p in all_purchases:
                p_datetime = p['datetime']
                # Ensure purchase datetime is timezone-aware for comparison
                if p_datetime.tzinfo is None:
                    p_datetime = tz.localize(p_datetime)
                if p_datetime <= burst_cutoff:
                    burst_purchases.append(p)
            
            if len(burst_purchases) == len(all_purchases) and days_since_first >= config.inactivity_days:
                return SpendingFrequency.IO
        
        # Count unique purchase days in window
        purchase_days = set()
        for p in window_purchases:
            p_datetime = p['datetime']
            # Ensure timezone-aware for date extraction
            if p_datetime.tzinfo is None:
                p_datetime = tz.localize(p_datetime)
            purchase_days.add(p_datetime.date())
        days_per_week = len(purchase_days) / 4  # 28-day window = 4 weeks
        
        if days_per_week >= 7:
            return SpendingFrequency.ED
        elif days_per_week >= 4:
            return SpendingFrequency.AD
        elif days_per_week >= 1:
            return SpendingFrequency.WS
        else:
            return SpendingFrequency.NW
    
    def _analyze_engagement_behavior(self, window_messages: List[Message],
                                   window_purchases: List[Dict],
                                   conv_stats: Dict) -> EngagementBehavior:
        """Analyze engagement with precedence: CB > Q > ER > L > CN."""
        fan_messages = [m for m in window_messages if m.sender_type.value == 'fan']
        message_count = len(fan_messages)
        has_purchases = len(window_purchases) > 0
        
        # CB: Chatty Buyer (20+ messages + purchases)
        if message_count >= 20 and has_purchases:
            return EngagementBehavior.CB
        
        # Q: Quick Responder (avg < 2 min)
        avg_response = conv_stats.get('avg_response_time_minutes', float('inf'))
        if avg_response is not None and avg_response < 2:
            return EngagementBehavior.Q
        
        # ER: Event-Driven (need to detect spikes - simplified for now)
        # TODO: Implement proper event correlation
        
        # L: Lurker (< 5 messages)
        if message_count < 5:
            return EngagementBehavior.L
        
        # CN: Chatty Non-Buyer (default for chatty without purchases)
        return EngagementBehavior.CN
    
    def _analyze_emotional_attachment(self, window_messages: List[Message],
                                    window_purchases: List[Dict],
                                    config: SegmentationConfig) -> EmotionalAttachment:
        """Analyze emotional attachment with confidence threshold."""
        # Calculate tip ratio for Gift-Giver detection
        total_spent = sum(p['price'] + p['tips'] for p in window_purchases)
        total_tips = sum(p['tips'] for p in window_purchases)
        tip_ratio = total_tips / total_spent if total_spent > 0 else 0
        
        # Check Gift-Giver first (dual threshold)
        if tip_ratio >= config.gift_giver_tip_ratio and total_tips >= config.gift_giver_min_tips:
            return EmotionalAttachment.G
        
        # Aggregate emotions from fan messages
        fan_messages = [m for m in window_messages if m.sender_type.value == 'fan' and m.emotions]
        if not fan_messages:
            return EmotionalAttachment.F  # Default to Friendly
        
        emotion_totals = Counter()
        emotion_counts = Counter()
        
        for msg in fan_messages:
            if msg.emotions:
                for emotion, score in msg.emotions.items():
                    emotion_totals[emotion] += score
                    emotion_counts[emotion] += 1
        
        # Calculate average scores
        emotion_avgs = {}
        for emotion, total in emotion_totals.items():
            emotion_avgs[emotion] = total / emotion_counts[emotion]
        
        # Determine dominant emotion with confidence
        if not emotion_avgs:
            return EmotionalAttachment.F
        
        sorted_emotions = sorted(emotion_avgs.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_emotions) < 2:
            return EmotionalAttachment.F
        
        # Check confidence (difference between top 2 emotions)
        confidence = sorted_emotions[0][1] - sorted_emotions[1][1]
        if confidence < config.emotion_min_confidence:
            return EmotionalAttachment.F
        
        # Map emotions to attachment types
        dominant = sorted_emotions[0][0]
        if dominant in ['love', 'joy'] and emotion_avgs.get('love', 0) > 0.5:
            return EmotionalAttachment.R
        elif dominant in ['sadness', 'fear'] or 'stress' in ' '.join(m.message.lower() for m in fan_messages):
            return EmotionalAttachment.E
        # TODO: Implement Collector detection based on purchase patterns
        
        return EmotionalAttachment.F
    
    def _analyze_lifecycle_stage(self, all_messages: List[Message],
                                all_purchases: List[Dict],
                                config: SegmentationConfig) -> LifecycleStage:
        """Analyze lifecycle stage based on account age and trends."""
        if not all_messages:
            return LifecycleStage.N
        
        first_msg_date = all_messages[0].datetime
        last_msg_date = all_messages[-1].datetime
        
        tz = self._get_timezone()
        # Handle timezone for first message
        if first_msg_date.tzinfo is None:
            first_msg_date = tz.localize(first_msg_date)
        # Handle timezone for last message  
        if last_msg_date.tzinfo is None:
            last_msg_date = tz.localize(last_msg_date)
            
        current_time = self._get_timezone_aware_now()
        account_age_days = (current_time - first_msg_date).days
        days_since_last = (current_time - last_msg_date).days
        
        # CH: Churned (>30 days inactive)
        if days_since_last > config.churn_days:
            return LifecycleStage.CH
        
        # N: New (<14 days)
        if account_age_days < 14:
            return LifecycleStage.N
        
        # Analyze recent trends (last 14 days vs previous 14 days)
        two_weeks_ago = self._get_timezone_aware_now() - timedelta(days=14)
        four_weeks_ago = self._get_timezone_aware_now() - timedelta(days=28)

        recent_msgs = [m for m in all_messages if self._to_tz(m.datetime) > two_weeks_ago]
        previous_msgs = [m for m in all_messages if four_weeks_ago <self._to_tz(m.datetime) <= two_weeks_ago]

        recent_purchases = [p for p in all_purchases if self._to_tz(p['datetime']) > two_weeks_ago]
        previous_purchases = [p for p in all_purchases if four_weeks_ago <self._to_tz(p['datetime']) <= two_weeks_ago]

        
        # Calculate activity change
        recent_activity = len(recent_msgs) + len(recent_purchases) * 5  # Weight purchases
        previous_activity = len(previous_msgs) + len(previous_purchases) * 5
        
        if previous_activity > 0:
            activity_change = (recent_activity - previous_activity) / previous_activity
            
            # W: Warming Up (>50% increase)
            if activity_change > 0.5:
                return LifecycleStage.W
            # CO: Cooling Off (>50% decrease)
            elif activity_change < -0.5:
                return LifecycleStage.CO
        
        # P: Peak (stable activity)
        return LifecycleStage.P
    
    def _analyze_ltv_segment(self, total_spent: float) -> LTVSegment:
        """Analyze LTV segment based on total spent."""
        if total_spent >= self.config.ltv_whale_threshold:
            return LTVSegment.W  # Whale
        elif total_spent >= self.config.ltv_dolphin_threshold:
            return LTVSegment.D  # Dolphin
        elif total_spent >= self.config.ltv_shark_threshold:
            return LTVSegment.S  # Shark
        elif total_spent >= self.config.ltv_fish_threshold:
            return LTVSegment.F  # Fish
        else:
            return LTVSegment.M  # Minnow
    
    def _track_segment_change(self, fan_id: str, model_name: str, 
                            segment: FanSegment, features: Dict):
        """Track segment changes in history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current segment
        cursor.execute("""
            SELECT segment_code FROM fan_segment_history 
            WHERE fan_id = ? AND model_name = ? AND valid_to IS NULL
            ORDER BY valid_from DESC LIMIT 1
        """, (fan_id, model_name))
        
        current = cursor.fetchone()
        
        # Only track if changed or new
        if not current or current[0] != segment.segment_code:
            # Close previous segment
            if current:
                cursor.execute("""
                    UPDATE fan_segment_history 
                    SET valid_to = ? 
                    WHERE fan_id = ? AND model_name = ? AND valid_to IS NULL
                """, (self._get_timezone_aware_now(), fan_id, model_name))
            
            # Determine the reason for change
            reason = "Segment updated" if current else "Initial segmentation"
            
            # Insert new segment with reason
            cursor.execute("""
                INSERT INTO fan_segment_history 
                (fan_id, model_name, segment_code, spending_frequency, 
                 engagement_behavior, emotional_attachment, lifecycle_stage, ltv_segment,
                 valid_from, version, reason, features, is_provisional)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fan_id, model_name, segment.segment_code,
                segment.spending_frequency.value,
                segment.engagement_behavior.value,
                segment.emotional_attachment.value,
                segment.lifecycle_stage.value,
                segment.ltv_segment.value if segment.ltv_segment else 'M',
                self._get_timezone_aware_now(),
                features.get('config_version', 1),
                reason,
                json.dumps(features),
                segment.is_provisional
            ))
            
            segment.last_change_at = self._get_timezone_aware_now()
            segment.change_reason = reason
        
        conn.commit()
        conn.close()
    
    def get_dashboard_stats(self) -> DashboardStats:
        """Get overall dashboard statistics."""
        processor = self.data_processor
        
        total_fans = processor.merged_df['fan_id'].nunique()
        active_fans = len(processor.get_active_fans(24))
        total_revenue = processor.merged_df['revenue'].sum() + processor.merged_df['tips'].sum()
        avg_fan_value = total_revenue / total_fans if total_fans > 0 else 0
        top_spenders = processor.get_top_spenders(5)
        
        mood_distribution = {
            MoodType.HAPPY: 0,
            MoodType.SAD: 0,
            MoodType.EXCITED: 0,
            MoodType.NEUTRAL: 0,
            MoodType.ANGRY: 0,
            MoodType.LOVING: 0
        }
        
        return DashboardStats(
            total_fans=total_fans,
            active_fans_today=active_fans,
            total_revenue=total_revenue,
            avg_fan_value=avg_fan_value,
            top_spenders=top_spenders,
            mood_distribution=mood_distribution
        )