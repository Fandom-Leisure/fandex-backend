# Fan Segmentation System Documentation

## Overview

The fan segmentation system uses a **5-axis classification model** to categorize fans based on their behavior, spending patterns, emotional attachment, lifecycle stage, and lifetime value. Each fan receives a segment code like `"WS-CB-R-P-D"` that represents their classification across all five dimensions.

## The 5-Axis Segmentation Model

### 1. Spending Frequency (First Axis)
Analyzes how often a fan makes purchases within a 28-day rolling window.

- **TW (Time-Waster)**: $0 spent
- **NW (Not Worth It)**: Below profitable threshold (<$50 in window)
- **IO (In and Out)**: Burst buyer who spent only in first 48 hours then went inactive (14+ days)
- **WS (Weekly Spender)**: 1-2 purchase days per week
- **AD (Almost Daily)**: 4-6 purchase days per week  
- **ED (Every Day)**: Daily purchases (7+ days per week)

### 2. Engagement Behavior (Second Axis)
Measures communication patterns and responsiveness.

**Priority order**: CB > Q > ER > L > CN

- **L (Lurker)**: Less than 5 messages in 28-day window
- **CB (Chatty Buyer)**: 20+ messages AND has purchases
- **CN (Chatty Non-Buyer)**: Chatty but no purchases
- **Q (Quick Responder)**: Average response time < 2 minutes
- **ER (Event-Driven)**: Active mainly after promotions/events

### 3. Emotional Attachment (Third Axis)
Analyzes the emotional connection type based on messages and tipping behavior.

- **R (Romantic)**: Dominant emotions are love/joy with love score > 0.5
- **F (Friendly)**: Default/neutral emotional state
- **C (Collector)**: Completes content sets (not yet implemented)
- **E (Escape-Seeker)**: Shows stress/sadness or mentions stress
- **G (Gift-Giver)**: Tips ≥30% of total spent AND tips ≥$100

### 4. Lifecycle Stage (Fourth Axis)
Tracks the fan's journey and current activity trend.

- **N (New)**: Account age < 14 days
- **W (Warming Up)**: Activity increased >50% vs previous 14 days
- **P (Peak)**: Stable activity pattern
- **CO (Cooling Off)**: Activity decreased >50% vs previous 14 days
- **CH (Churned)**: No activity for 30+ days

### 5. LTV Segment (Fifth Axis)
Categorizes by total lifetime spending.

- **W (Whale)**: $1,000+ total spent
- **S (Shark)**: $500-999 total spent
- **F (Fish)**: $100-499 total spent
- **M (Minnow)**: <$100 total spent

## Segmentation Process

### 1. Data Collection
- Uses a **28-day rolling window** for recent behavior analysis
- Collects all messages and purchases within the window
- Maintains full history for lifecycle and LTV analysis

### 2. Provisional Status
- Fans with account age < 14 days are marked as **provisional**
- Provisional segments may change rapidly as more data is collected

### 3. Computation Flow

```python
def _compute_fan_segmentation():
    1. Get 28-day window data (messages & purchases)
    2. Check if fan is provisional (<14 days old)
    3. Analyze each axis independently:
       - Spending Frequency: Count unique purchase days
       - Engagement: Check message count & response times
       - Emotional: Aggregate emotion scores & tip ratios
       - Lifecycle: Compare recent vs previous activity
       - LTV: Sum total lifetime spending
    4. Generate segment code: "SF-EB-EA-LS-LTV"
    5. Track segment changes in history database
```

### 4. Key Thresholds (Configurable)

- **Profitable threshold**: $50 (minimum for non-NW classification)
- **Gift-giver tip ratio**: 30% of total spent
- **Gift-giver minimum tips**: $100
- **Emotion confidence**: 0.7 (difference between top 2 emotions)
- **Burst window**: 48 hours
- **Inactivity period**: 14 days
- **Churn period**: 30 days

### 5. Timezone Handling
- All calculations use creator's timezone (default: Asia/Manila)
- Ensures consistent day boundaries for purchase frequency
- Latest data timestamp used as "now" reference

## Segment History Tracking

The system maintains a complete history of segment changes:

- **When segments change**: New record created with reason
- **Provisional segments**: Marked with `is_provisional=true`
- **Audit trail**: Stores computation features for debugging
- **Version control**: Tracks configuration version used

## Example Segment Codes

- `"WS-CB-R-P-D"`: Weekly spender, chatty buyer, romantic, at peak, dolphin-level LTV
- `"TW-L-F-N-M"`: Time-waster, lurker, friendly, new fan, minnow-level LTV
- `"ED-Q-G-P-W"`: Daily spender, quick responder, gift-giver, at peak, whale-level LTV
- `"IO-CN-E-CH-F"`: In-and-out spender, chatty non-buyer, escape-seeker, churned, fish-level LTV

## Business Applications

1. **Personalized Messaging**: Tailor content based on emotional attachment
2. **Timing Optimization**: Use lifecycle stage to time offers
3. **Value Prediction**: LTV segment guides resource allocation
4. **Retention Strategies**: Target CO (cooling off) fans before they churn
5. **Upsell Opportunities**: Identify WS fans ready to become AD

## Technical Implementation

The segmentation is computed in real-time when analyzing a fan through:
- `FanAnalyzer._compute_fan_segmentation()` (main method)
- Individual axis analyzers: `_analyze_spending_frequency()`, etc.
- Results stored in `FanSegment` model with 5 enum fields
- History tracked in SQLite database table `fan_segment_history`