const API_BASE = 'http://localhost:8000/api';

// Global state
let currentViewMode = 'fans';

// Load dashboard stats on page load
document.addEventListener('DOMContentLoaded', () => {
    loadDashboardStats();
    loadModels();
    loadFans();
});

async function loadDashboardStats() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const stats = await response.json();
        
        document.getElementById('totalFans').textContent = stats.total_fans.toLocaleString();
        document.getElementById('activeFans').textContent = stats.active_fans_today.toLocaleString();
        document.getElementById('totalRevenue').textContent = `$${stats.total_revenue.toFixed(2)}`;
        document.getElementById('avgFanValue').textContent = `$${stats.avg_fan_value.toFixed(2)}`;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function loadModels() {
    try {
        const response = await fetch(`${API_BASE}/models`);
        const data = await response.json();
        const select = document.getElementById('modelFilter');
        
        // Clear existing options except the first one
        select.innerHTML = '<option value="">All Models</option>';
        
        // Add model options
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

async function loadFans(offset = 0, search = '') {
    const grid = document.getElementById('fansGrid');
    grid.innerHTML = '<div class="text-center py-10 text-gray-400">Loading fans...</div>';
    
    const modelFilter = document.getElementById('modelFilter');
    const selectedModel = modelFilter.value;
    
    // Get all filter values
    const spendingFilter = Array.from(document.getElementById('spendingFilter').selectedOptions).map(opt => opt.value);
    const engagementFilter = Array.from(document.getElementById('engagementFilter').selectedOptions).map(opt => opt.value);
    const emotionalFilter = Array.from(document.getElementById('emotionalFilter').selectedOptions).map(opt => opt.value);
    const lifecycleFilter = Array.from(document.getElementById('lifecycleFilter').selectedOptions).map(opt => opt.value);
    const ltvFilter = Array.from(document.getElementById('ltvFilter').selectedOptions).map(opt => opt.value);
    
    try {
        let url = `${API_BASE}/fans?limit=20&offset=${offset}`;
        if (search) {
            url += `&search=${encodeURIComponent(search)}`;
        }
        if (currentViewMode === 'fan-model') {
            url += '&group_by_model=true';
            if (selectedModel) {
                url += `&model_name=${encodeURIComponent(selectedModel)}`;
            }
        }
        
        // Add filter parameters
        if (spendingFilter.length > 0) {
            url += `&spending_frequency=${spendingFilter.join(',')}`;
        }
        if (engagementFilter.length > 0) {
            url += `&engagement_behavior=${engagementFilter.join(',')}`;
        }
        if (emotionalFilter.length > 0) {
            url += `&emotional_attachment=${emotionalFilter.join(',')}`;
        }
        if (lifecycleFilter.length > 0) {
            url += `&lifecycle_stage=${lifecycleFilter.join(',')}`;
        }
        if (ltvFilter.length > 0) {
            url += `&ltv_segment=${ltvFilter.join(',')}`;
        }
        
        const response = await fetch(url);
        const fans = await response.json();
        
        grid.innerHTML = '';
        
        for (const fan of fans) {
            if (currentViewMode === 'fan-model') {
                // For fan-model pairs, load with specific model
                const summary = await loadFanSummary(fan.fan_id, fan.model_name);
                if (summary) {
                    grid.appendChild(createFanCard(summary, fan));
                }
            } else {
                // For fans only view
                const summary = await loadFanSummary(fan.fan_id);
                if (summary) {
                    grid.appendChild(createFanCard(summary, fan));
                }
            }
        }
    } catch (error) {
        grid.innerHTML = '<div class="bg-red-500 text-white p-4 rounded-lg m-5">Error loading fans. Please try again.</div>';
        console.error('Error loading fans:', error);
    }
}

async function loadFanSummary(fanId, modelName = null) {
    try {
        let url = `${API_BASE}/fans/${fanId}`;
        if (modelName) {
            url += `?model_name=${encodeURIComponent(modelName)}`;
        }
        const response = await fetch(url);
        return await response.json();
    } catch (error) {
        console.error(`Error loading fan ${fanId}:`, error);
        return null;
    }
}

function createFanCard(fan, listData = null) {
    const template = document.getElementById('fanCardTemplate').innerHTML;
    
    // Determine value color based on total spent
    let valueColor = 'bg-gray-600 text-white';
    if (fan.total_spent > 100) valueColor = 'bg-green-500 text-white';
    else if (fan.total_spent > 50) valueColor = 'bg-yellow-500 text-white';
    
    // Determine readiness color
    let readinessColor = 'bg-red-500 text-white';
    if (fan.purchase_readiness === 'high') readinessColor = 'bg-green-500 text-white';
    else if (fan.purchase_readiness === 'medium') readinessColor = 'bg-yellow-500 text-white';
    
    // Format recent topics
    const recentTopicsHtml = fan.recent_topics
        .map(topic => `<span class="px-3 py-1 bg-gray-800 rounded-full text-xs text-gray-400">${topic}</span>`)
        .join('');
    
    // Handle model display
    let modelDisplay = fan.model_name || 'Unknown';
    if (listData && listData.model_names && Array.isArray(listData.model_names)) {
        if (listData.model_names.length > 1) {
            modelDisplay = `${listData.model_names[0]} (+${listData.model_names.length - 1} more)`;
        } else {
            modelDisplay = listData.model_names[0] || 'Unknown';
        }
    }
    
    // Handle segmentation display
    let segmentCode = 'N/A';
    let segmentColor = 'bg-gray-600';
    let provisionalBadge = '';
    let ltvBadge = '';
    
    if (fan.segmentation) {
        segmentCode = fan.segmentation.segment_code;
        
        // Color based on spending frequency
        const spending = fan.segmentation.spending_frequency;
        if (spending === 'ED' || spending === 'AD') {
            segmentColor = 'bg-green-600';
        } else if (spending === 'WS') {
            segmentColor = 'bg-yellow-600';
        } else if (spending === 'IO' || spending === 'NW') {
            segmentColor = 'bg-orange-600';
        } else {
            segmentColor = 'bg-red-600';
        }
        
        // Add LTV badge with gaming-themed styling
        if (fan.segmentation.ltv_segment) {
            const ltvColors = {
                'W': 'bg-gradient-to-r from-purple-600 to-yellow-500',  // Whale - Purple/Gold gradient
                'D': 'bg-blue-600',                                     // Dolphin - Blue
                'S': 'bg-gray-800',                                     // Shark - Dark gray
                'F': 'bg-gray-500',                                     // Fish - Silver
                'M': 'bg-amber-700'                                     // Minnow - Bronze
            };
            const ltvLabels = {
                'W': '🐋 Whale',
                'D': '🐬 Dolphin', 
                'S': '🦈 Shark',
                'F': '🐟 Fish',
                'M': '🐠 Minnow'
            };
            const ltvColor = ltvColors[fan.segmentation.ltv_segment] || 'bg-gray-600';
            const ltvLabel = ltvLabels[fan.segmentation.ltv_segment] || fan.segmentation.ltv_segment;
            ltvBadge = `<span class="text-xs ${ltvColor} text-white px-2 py-1 rounded ml-1 font-semibold">${ltvLabel}</span>`;
        }
        
        if (fan.segmentation.is_provisional) {
            provisionalBadge = '<span class="text-xs bg-purple-600 text-white px-2 py-1 rounded ml-1">NEW</span>';
        }
    } else if (listData && listData.segment_code) {
        segmentCode = listData.segment_code;
        if (listData.is_provisional) {
            provisionalBadge = '<span class="text-xs bg-purple-600 text-white px-2 py-1 rounded ml-1">NEW</span>';
        }
    }
    
    // Format successful topics
    const topicsText = fan.successful_topics.join(', ');
    
    // Truncate last message
    const lastMessage = fan.last_message ? 
        (fan.last_message.length > 50 ? fan.last_message.substring(0, 50) + '...' : fan.last_message) : 
        'No messages yet';
    
    // Replace all placeholders
    let html = template
        .replace(/{fan_id}/g, fan.fan_id)
        .replace(/{model_name}/g, fan.model_name || '')
        .replace(/{total_spent}/g, fan.total_spent.toFixed(2))
        .replace(/{value_color}/g, valueColor)
        .replace(/{days_since}/g, fan.days_since_last_purchase || 'Never')
        .replace(/{last_message}/g, lastMessage)
        .replace(/{response_pattern}/g, fan.response_pattern)
        .replace(/{streak}/g, fan.conversation_streak)
        .replace(/{best_time}/g, fan.best_time_to_chat)
        .replace(/{message_style}/g, fan.preferred_message_style)
        .replace(/{topics}/g, topicsText)
        .replace(/{action_suggestion}/g, fan.next_action_suggestion)
        .replace(/{mood_emoji}/g, fan.mood_emoji)
        .replace(/{mood_text}/g, fan.current_mood)
        .replace(/{readiness_color}/g, readinessColor)
        .replace(/{readiness_score}/g, fan.purchase_readiness_score)
        .replace(/{readiness_text}/g, fan.purchase_readiness)
        .replace(/{notes}/g, fan.personal_notes)
        .replace(/{recent_topics_html}/g, recentTopicsHtml)
        .replace(/{model_display}/g, modelDisplay)
        .replace(/{segment_code}/g, segmentCode)
        .replace(/{segment_color}/g, segmentColor)
        .replace(/{provisional_badge}/g, provisionalBadge)
        .replace(/{ltv_badge}/g, ltvBadge);
    
    const div = document.createElement('div');
    div.innerHTML = html;
    return div.firstElementChild;
}

async function searchFans() {
    const searchTerm = document.getElementById('fanSearch').value;
    await loadFans(0, searchTerm);
}

async function loadActiveFans() {
    const grid = document.getElementById('fansGrid');
    grid.innerHTML = '<div class="text-center py-10 text-gray-400">Loading active fans...</div>';
    
    try {
        const response = await fetch(`${API_BASE}/active-fans?hours=24`);
        const activeFans = await response.json();
        
        grid.innerHTML = '';
        
        for (const fan of activeFans) {
            const summary = await loadFanSummary(fan.fan_id);
            if (summary) {
                grid.appendChild(createFanCard(summary));
            }
        }
        
        if (activeFans.length === 0) {
            grid.innerHTML = '<div class="text-center py-10 text-gray-400">No active fans in the last 24 hours</div>';
        }
    } catch (error) {
        grid.innerHTML = '<div class="bg-red-500 text-white p-4 rounded-lg m-5">Error loading active fans. Please try again.</div>';
        console.error('Error loading active fans:', error);
    }
}

async function loadFanDetails(fanId, modelName = null) {
    const modal = document.getElementById('fanModal');
    const detailDiv = document.getElementById('fanDetail');
    
    modal.classList.remove('hidden');
    detailDiv.innerHTML = '<div class="text-center py-10 text-gray-400">Loading fan details...</div>';
    
    try {
        let summaryUrl = `${API_BASE}/fans/${fanId}`;
        let historyUrl = `${API_BASE}/fans/${fanId}/history?limit=50`;
        let conversationsUrl = `${API_BASE}/fans/${fanId}/conversations?limit=5`;
        
        if (modelName && modelName !== 'null' && modelName !== 'undefined') {
            summaryUrl += `?model_name=${encodeURIComponent(modelName)}`;
            historyUrl += `&model_name=${encodeURIComponent(modelName)}`;
            conversationsUrl += `&model_name=${encodeURIComponent(modelName)}`;
        }
        
        const [summary, history, conversations] = await Promise.all([
            fetch(summaryUrl).then(r => r.json()),
            fetch(historyUrl).then(r => r.json()),
            fetch(conversationsUrl).then(r => r.json())
        ]);
        
        detailDiv.innerHTML = `
            <h2 class="text-2xl font-bold mb-6">${fanId}</h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                <div class="bg-gray-900 p-4 rounded-lg">
                    <div class="text-gray-400 text-sm">Total Spent</div>
                    <div class="text-xl font-bold text-green-400">$${summary.total_spent.toFixed(2)}</div>
                </div>
                <div class="bg-gray-900 p-4 rounded-lg">
                    <div class="text-gray-400 text-sm">Messages</div>
                    <div class="text-xl font-bold">${summary.message_count}</div>
                </div>
                <div class="bg-gray-900 p-4 rounded-lg">
                    <div class="text-gray-400 text-sm">Conversations</div>
                    <div class="text-xl font-bold">${summary.total_conversations}</div>
                </div>
                <div class="bg-gray-900 p-4 rounded-lg">
                    <div class="text-gray-400 text-sm">Current Mood</div>
                    <div class="text-xl">${summary.mood_emoji} ${summary.current_mood}</div>
                </div>
            </div>
            
            <h3 class="text-xl font-semibold mb-4">Recent Conversations</h3>
            <div class="space-y-3 mb-8">
                ${conversations.map(conv => `
                    <div class="bg-gray-900 p-4 rounded-lg">
                        <div class="text-sm text-gray-400">${new Date(conv.start_time).toLocaleDateString()}</div>
                        <div class="text-gray-300 mt-1">${conv.preview}</div>
                        <div class="text-xs text-gray-500 mt-2">${conv.message_count} messages</div>
                    </div>
                `).join('')}
            </div>
            
            <h3 class="text-xl font-semibold mb-4">Purchase History</h3>
            <div class="space-y-2">
                ${history.purchase_history.map(purchase => `
                    <div class="flex justify-between items-center bg-gray-900 p-3 rounded-lg">
                        <div class="text-gray-400">${new Date(purchase.datetime).toLocaleDateString()}</div>
                        <div class="text-green-400 font-bold">$${purchase.total.toFixed(2)}</div>
                    </div>
                `).join('') || '<p class="text-gray-500">No purchases yet</p>'}
            </div>
        `;
    } catch (error) {
        detailDiv.innerHTML = '<div class="bg-red-500 text-white p-4 rounded-lg">Error loading fan details. Please try again.</div>';
        console.error('Error loading fan details:', error);
    }
}

function closeFanModal() {
    document.getElementById('fanModal').classList.add('hidden');
}

async function saveNotes(fanId) {
    const notesInput = document.getElementById(`notes-${fanId}`);
    const notes = notesInput.value;
    
    try {
        await fetch(`${API_BASE}/fans/${fanId}/notes`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ notes })
        });
    } catch (error) {
        console.error('Error saving notes:', error);
    }
}

// Handle view mode change
function handleViewModeChange() {
    const viewMode = document.querySelector('input[name="viewMode"]:checked').value;
    currentViewMode = viewMode;
    
    const modelFilter = document.getElementById('modelFilter');
    if (viewMode === 'fan-model') {
        modelFilter.classList.remove('hidden');
    } else {
        modelFilter.classList.add('hidden');
    }
    
    loadFans();
}

// Handle model filter change
document.getElementById('modelFilter')?.addEventListener('change', () => {
    if (currentViewMode === 'fan-model') {
        loadFans();
    }
});

// Handle search on Enter key
document.getElementById('fanSearch')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        searchFans();
    }
});

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('fanModal');
    if (event.target === modal) {
        closeFanModal();
    }
}