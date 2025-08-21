# Fan Intelligence Dashboard MVP+

A comprehensive dashboard for analyzing fan engagement, behavior, and purchase patterns with 8 essential features for optimizing fan interactions.

## Features

### Core Features (1-4)
1. **Fan Value Indicator** - Color-coded badges showing total spent and days since last purchase
2. **Recent Activity Summary** - Last message preview, response patterns, and conversation streaks
3. **What Works With This Fan** - Best chat times, preferred message styles, and successful topics
4. **Quick Action Suggestions** - AI-powered recommendations for the next best action

### Additional Features (5-8)
5. **Mood Tracker** - Real-time emotional state analysis with emoji indicators
6. **Purchase Readiness Score** - 1-10 scale showing likelihood to make a purchase
7. **Personal Notes Section** - Persistent notes for building personal connections
8. **Conversation History Quick View** - Recent topics to avoid repetition and maintain engagement

## Installation

1. **Install Dependencies**
   ```bash
   cd dashboard
   pip install -r requirements.txt
   ```

2. **Ensure Data Files Exist**
   The dashboard requires these files in the `data/` directory:
   - `all_chatlogs_message_level.pkl`
   - `messages_with_emotions.pkl`

## Running the Dashboard

1. **From the dashboard directory:**
   ```bash
   python run_dashboard.py
   ```

2. **Or manually start the servers:**
   ```bash
   # Terminal 1 - Backend
   cd dashboard
   uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
   
   # Terminal 2 - Frontend
   cd dashboard
   python -m http.server 8080 --directory frontend
   ```

3. **Access the dashboard:**
   - Frontend: http://localhost:8080
   - API Docs: http://localhost:8000/docs

## Usage Guide

### Dashboard Overview
- **Header Stats**: View total fans, active fans today, total revenue, and average fan value
- **Search**: Find specific fans by ID
- **Active Fans**: Quick filter to show only recently active fans

### Fan Cards
Each fan card displays all 8 features in an easy-to-scan format:
- Click any card to view detailed history
- Edit personal notes directly on the card
- Color coding indicates fan value and purchase readiness

### Understanding the Indicators

#### Fan Value Colors
- 🟢 **Green**: High value ($100+)
- 🟡 **Yellow**: Medium value ($50-$100)
- ⚫ **Gray**: Low value (< $50)

#### Purchase Readiness
- 🟢 **Green (8-10)**: High - Perfect time for offers
- 🟡 **Yellow (5-7)**: Medium - Build more rapport
- 🔴 **Red (1-4)**: Low - Focus on engagement

#### Mood Emojis
- 😊 Happy
- 😔 Sad
- 🔥 Excited
- 😐 Neutral
- 😠 Angry
- 🥰 Loving

## API Endpoints

- `GET /api/fans` - List all fans with pagination
- `GET /api/fans/{fan_id}` - Get detailed fan summary
- `GET /api/fans/{fan_id}/history` - Get message history
- `POST /api/fans/{fan_id}/notes` - Update personal notes
- `GET /api/stats` - Get dashboard statistics
- `GET /api/active-fans` - Get recently active fans

## Architecture

### Backend
- **FastAPI** for high-performance API
- **Pandas** for efficient data processing
- **SQLite** for persistent notes storage
- **Pydantic** for data validation

### Frontend
- **Vanilla JavaScript** for maintainability
- **Responsive CSS Grid** layout
- **Real-time updates** via API polling
- **Modal system** for detailed views

## Data Processing

The dashboard efficiently handles 8.5M+ messages by:
- Lazy loading data on demand
- Caching frequently accessed fan data
- Aggregating statistics at runtime
- Merging emotion data only when needed

## Customization

### Adding New Features
1. Update `models.py` with new fields
2. Add analysis logic to `fan_analyzer.py`
3. Create new API endpoint in `api.py`
4. Update frontend card template in `index.html`

### Styling
Edit `styles.css` to customize:
- Color scheme (CSS variables)
- Card layouts
- Responsive breakpoints

## Troubleshooting

### Common Issues

1. **"Required data file not found"**
   - Ensure pickle files are in the `data/` directory
   - Check file permissions

2. **"Port already in use"**
   - Change ports in `run_dashboard.py`
   - Kill existing processes on ports 8000/8080

3. **Slow loading**
   - First load processes all data
   - Subsequent loads use cache
   - Consider reducing initial fan limit

## Performance Tips

- Start with "Show Active Fans" for faster initial load
- Use search for specific fans instead of scrolling
- Keep browser console open to monitor API calls
- Close detailed views when not needed

## Future Enhancements

- WebSocket support for real-time updates
- Advanced filtering and sorting
- Bulk actions for multiple fans
- Export functionality
- Mobile app version