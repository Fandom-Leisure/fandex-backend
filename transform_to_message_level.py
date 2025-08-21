import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# First, we need to identify sessions based on 30-minute gaps
def identify_sessions(df):
    """Identify session IDs based on 30-minute gaps between messages (vectorized)"""
    print("Sorting data for session identification...")
    # Sort by fan_id, model_name, and datetime
    df_sorted = df.sort_values(['fan_id', 'model_name', 'datetime']).copy()
    
    # Calculate time differences within each fan-model group
    print("Calculating time gaps...")
    df_sorted['time_diff'] = df_sorted.groupby(['fan_id', 'model_name'])['datetime'].diff()
    
    # Mark new sessions (True when gap > 30 minutes or first message)
    time_threshold = pd.Timedelta(minutes=30)
    df_sorted['new_session'] = (df_sorted['time_diff'] > time_threshold) | df_sorted['time_diff'].isna()
    
    # Assign session numbers within each fan-model group
    print("Assigning session IDs...")
    df_sorted['session_id'] = df_sorted.groupby(['fan_id', 'model_name'])['new_session'].cumsum()
    
    # Drop temporary columns
    df_sorted = df_sorted.drop(['time_diff', 'new_session'], axis=1)
    
    # Map session_ids back to original dataframe order
    session_mapping = pd.Series(df_sorted['session_id'].values, index=df_sorted.index)
    df['session_id'] = session_mapping.loc[df.index]
    
    return df

# Vectorized approach for better performance
def transform_to_messages(df_chunk):
    # Create arrays for fan messages
    fan_mask = df_chunk['fan_message'].notna() & (df_chunk['fan_message'] != '')
    fan_data = {
        'chatter_name': df_chunk['chatter_name'][fan_mask],
        'model_name': df_chunk['model_name'][fan_mask],
        'fan_id': df_chunk['fan_id'][fan_mask],
        'datetime': df_chunk['datetime'][fan_mask],
        'sender_type': 'fan',
        'sender_id': df_chunk['fan_id'][fan_mask],
        'message': df_chunk['fan_message'][fan_mask],
        'price': 0.0,
        'purchased': False,
        'tips': df_chunk['tips'][fan_mask].fillna(0.0),
        'revenue': 0.0,
        'session_id': df_chunk['session_id'][fan_mask]
    }
    
    # Create arrays for chatter messages
    chatter_mask = df_chunk['chatter_message'].notna() & (df_chunk['chatter_message'] != '')
    chatter_data = {
        'chatter_name': df_chunk['chatter_name'][chatter_mask],
        'model_name': df_chunk['model_name'][chatter_mask],
        'fan_id': df_chunk['fan_id'][chatter_mask],
        'datetime': df_chunk['datetime'][chatter_mask],
        'sender_type': 'chatter',
        'sender_id': df_chunk['chatter_name'][chatter_mask],
        'message': df_chunk['chatter_message'][chatter_mask],
        'price': df_chunk['price'][chatter_mask].fillna(0.0),
        'purchased': df_chunk['purchased'][chatter_mask].fillna(False),
        'tips': 0.0,
        'revenue': df_chunk['revenue'][chatter_mask].fillna(0.0),
        'session_id': df_chunk['session_id'][chatter_mask]
    }
    
    # Create DataFrames
    fan_df = pd.DataFrame(fan_data)
    chatter_df = pd.DataFrame(chatter_data)
    
    # Combine
    combined_df = pd.concat([fan_df, chatter_df], ignore_index=True)
    
    # Create conversation_id as fan-id_model-name_session-id
    combined_df['conversation_id'] = combined_df['fan_id'].astype(str) + '_' + \
                                     combined_df['model_name'].astype(str) + '_' + \
                                     combined_df['session_id'].astype(str)
    
    return combined_df

# Load the pickle file
print("Loading data...")
with open('data/all_chatlogs.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"Original data shape: {df.shape}")
print(f"Estimated memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# First, identify sessions for the entire dataset
print("\nIdentifying sessions based on 30-minute gaps...")
df = identify_sessions(df)

# First, let's process a sample to verify the transformation
print("\nProcessing sample data first...")
sample_df = df.head(1000)

# Test with sample
sample_messages = transform_to_messages(sample_df)
print(f"\nSample transformation complete. Shape: {sample_messages.shape}")
print("Sample data:")
print(sample_messages.head(10))

# Process full dataset in chunks
print("\nProcessing full dataset in chunks...")
chunk_size = 100000
message_dfs = []

for i in range(0, len(df), chunk_size):
    chunk_end = min(i + chunk_size, len(df))
    print(f"Processing rows {i:,} to {chunk_end:,}...")
    
    chunk = df.iloc[i:chunk_end]
    chunk_messages = transform_to_messages(chunk)
    message_dfs.append(chunk_messages)

# Combine all chunks
print("\nCombining all message chunks...")
message_df = pd.concat(message_dfs, ignore_index=True)

# Sort by conversation_id and datetime
print("Sorting messages...")
message_df = message_df.sort_values(['conversation_id', 'datetime']).reset_index(drop=True)


print(f"\nFinal message-level data shape: {message_df.shape}")
print(f"Memory usage: {message_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Show distribution
print("\nSender type distribution:")
print(message_df['sender_type'].value_counts())

# Show sample of final data
print("\nSample of transformed data:")
print(message_df.head(20))

# Show session statistics
print("\nSession statistics:")
unique_sessions = message_df['conversation_id'].nunique()
print(f"Total unique sessions: {unique_sessions:,}")

# Show sample conversation
print("\nSample conversation (first session):")
first_conversation = message_df['conversation_id'].iloc[0]
sample_conv = message_df[message_df['conversation_id'] == first_conversation]
print(sample_conv[['datetime', 'sender_type', 'message', 'conversation_id']].head(10))

# Save to new pickle file
output_file = 'data/all_chatlogs_message_level.pkl'
print(f"\nSaving to {output_file}...")
with open(output_file, 'wb') as f:
    pickle.dump(message_df, f)

print("Transformation complete!")