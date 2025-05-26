import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Constants
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NEGATIVE_EMOTIONS = ['angry', 'disgust', 'fear', 'sad']

# Visualization utilities
def generate_emotion_chart(emotion_history):
    """Generate a bar chart visualization of emotion distribution"""
    if not emotion_history:
        return None
    
    counts = Counter(emotion_history)
    df = pd.DataFrame({'emotion': list(counts.keys()), 'count': list(counts.values())})
    
    # Sort by emotions in a logical order
    emotion_order = ['happy', 'neutral', 'sad', 'angry', 'fear', 'disgust', 'surprise']
    df['order'] = df['emotion'].map({emotion: idx for idx, emotion in enumerate(emotion_order)})
    df = df.sort_values('order').drop('order', axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4CAF50' if emotion not in NEGATIVE_EMOTIONS else '#F44336' for emotion in df['emotion']]
    sns.barplot(x='emotion', y='count', data=df, palette=colors, ax=ax)
    ax.set_title('Emotion Distribution', fontsize=16)
    ax.set_xlabel('Emotion', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    
    return fig

def generate_emotion_time_series(emotion_history):
    """Generate a time series chart of emotions over session"""
    if not emotion_history or len(emotion_history) < 5:
        return None
        
    # Convert emotions to numeric values for visualization
    emotion_values = {
        'happy': 3,
        'neutral': 2,
        'surprise': 1,
        'fear': -1,
        'sad': -2,
        'disgust': -2.5,
        'angry': -3
    }
    
    # Create a dataframe for plotting
    values = [emotion_values.get(e, 0) for e in emotion_history]
    df = pd.DataFrame({
        'time_point': range(len(emotion_history)),
        'emotion': emotion_history,
        'value': values,
    })
    
    # Moving average to smooth the curve
    window_size = min(5, max(2, len(emotion_history) // 10))
    df['smoothed'] = df['value'].rolling(window=window_size, center=True).mean()
    df['smoothed'] = df['smoothed'].fillna(df['value'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='time_point', y='smoothed', data=df, ax=ax, linewidth=2)
    
    # Color regions based on emotion valence
    ax.axhspan(0, 4, color='#E8F5E9', alpha=0.3)  # Positive emotion zone
    ax.axhspan(-4, 0, color='#FFEBEE', alpha=0.3)  # Negative emotion zone
    
    ax.set_title('Emotional State Over Time', fontsize=16)
    ax.set_xlabel('Time (frames)', fontsize=14)
    ax.set_ylabel('Emotional Valence', fontsize=14)
    
    # Custom y-ticks for readability
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticklabels(['Angry', 'Sad', 'Fear', 'Neutral', 'Surprise', 'Neutral', 'Happy'])
    
    return fig

def generate_risk_gauge(risk_score, max_score=21):
    """Generate a gauge chart for risk visualization"""
    # Create risk levels
    if risk_score < 5:
        color = '#4CAF50'  # Green
        level = 'Minimal'
    elif risk_score < 10:
        color = '#FFC107'  # Amber
        level = 'Mild'
    elif risk_score < 15:
        color = '#FF9800'  # Orange
        level = 'Moderate'
    else:
        color = '#F44336'  # Red
        level = 'High'

    # Create a half-circle gauge
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})
    
    # Plot the gauge background
    theta = np.linspace(0, np.pi, 100)
    r = [0.8] * 100
    ax.plot(theta, r, color='lightgray', linewidth=20, alpha=0.3)
    
    # Calculate the position on the gauge based on the risk score
    gauge_position = (risk_score / max_score) * np.pi
    theta_filled = np.linspace(0, gauge_position, 100)
    r_filled = [0.8] * 100
    ax.plot(theta_filled, r_filled, color=color, linewidth=20)
    
    # Add a marker at the current position
    ax.scatter(gauge_position, 0.8, s=300, color=color, zorder=5, edgecolor='white')
    
    # Customize the plot
    ax.set_rticks([])  # Remove radial ticks
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])  # Set custom angular ticks
    ax.set_xticklabels(['0', '5', '10', '15', '21'])  # Label them with score values
    ax.spines['polar'].set_visible(False)  # Hide the circular spine
    
    # Add text annotations
    plt.text(np.pi/2, 0.4, f"{risk_score}/{max_score}", ha='center', va='center', fontsize=24, fontweight='bold')
    plt.text(np.pi/2, 0.2, f"{level} Risk", ha='center', va='center', fontsize=16)
    
    # Set the background color to white
    ax.set_facecolor('white')
    
    return fig

# Preprocessing utilities
def preprocess_face(face):
    """Preprocess a face image for the FER model"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(face).unsqueeze(0)

# Analysis utilities
def analyze_emotion_patterns(emotion_history):
    """Analyze emotion patterns and generate insights"""
    if not emotion_history:
        return None, 0
        
    counts = Counter(emotion_history)
    total = len(emotion_history)
    insights = []
    
    # Calculate emotion percentages
    emotion_percentages = {emotion: (count / total) * 100 for emotion, count in counts.items()}
    
    # Analyze negative emotions
    negative_count = sum(counts.get(emotion, 0) for emotion in NEGATIVE_EMOTIONS)
    negative_percentage = (negative_count / total) * 100 if total > 0 else 0
    
    # Generate insights based on patterns
    if negative_percentage > 70:
        insights.append("High negative emotion pattern detected. Shows significant distress.")
    elif negative_percentage > 50:
        insights.append("Moderate negative emotion pattern. May indicate emotional challenges.")
    
    # Check for emotional volatility (frequent changes between emotions)
    emotion_changes = sum(1 for i in range(1, len(emotion_history)) if emotion_history[i] != emotion_history[i-1])
    change_rate = emotion_changes / (len(emotion_history) - 1) if len(emotion_history) > 1 else 0
    
    if change_rate > 0.5:
        insights.append("High emotional volatility detected. May indicate emotional instability.")
    
    # Check for specific emotion patterns
    if emotion_percentages.get('angry', 0) > 30:
        insights.append("Significant anger expression may indicate frustration or hostility.")
    
    if emotion_percentages.get('sad', 0) > 30:
        insights.append("High sadness levels may indicate depressive tendencies.")
    
    if emotion_percentages.get('fear', 0) > 30:
        insights.append("Elevated fear response may indicate anxiety.")
        
    if emotion_percentages.get('happy', 0) > 70:
        insights.append("Very positive emotional state. No signs of emotional distress.")
        
    # If no specific insights were generated
    if not insights:
        if negative_percentage > 30:
            insights.append("Some negative emotions present but no clear pattern identified.")
        else:
            insights.append("Predominantly neutral or positive emotional state.")
    
    # Convert negative percentage to a risk score (0-10 scale)
    risk_score_from_emotion = min(10, round(negative_percentage / 10))
    
    return insights, risk_score_from_emotion

def calculate_combined_risk(questionnaire_score, emotion_risk_score):
    """Calculate a combined risk score from questionnaire and emotion analysis"""
    # Normalize questionnaire score to 0-10 scale
    normalized_questionnaire = min(10, questionnaire_score * 10 / 21)
    
    # Weighted average (giving more weight to questionnaire)
    combined_risk = (normalized_questionnaire * 0.7) + (emotion_risk_score * 0.3)
    
    # Map to risk categories
    if combined_risk < 3:
        return "Minimal Risk", combined_risk
    elif combined_risk < 5:
        return "Mild Risk", combined_risk
    elif combined_risk < 7:
        return "Moderate Risk", combined_risk
    else:
        return "High Risk", combined_risk

def get_coping_strategies(risk_level, emotion_insights):
    """Generate tailored coping strategies based on risk level and emotion insights"""
    general_strategies = [
        "Practice regular deep breathing exercises (5 minutes, 3 times daily)",
        "Maintain a consistent sleep schedule",
        "Engage in regular physical activity (at least 30 minutes daily)",
        "Limit caffeine and alcohol consumption",
        "Connect with supportive friends or family members"
    ]
    
    specific_strategies = {
        "Minimal Risk": [
            "Continue your current wellness practices",
            "Consider keeping a mood journal to track emotional patterns",
            "Schedule regular check-ins with healthcare provider"
        ],
        "Mild Risk": [
            "Implement daily mindfulness meditation (10-15 minutes)",
            "Consider mental health apps for daily mood tracking",
            "Schedule a follow-up assessment in 2-4 weeks"
        ],
        "Moderate Risk": [
            "Schedule a consultation with a mental health professional",
            "Increase self-care activities and social support",
            "Consider structured relaxation techniques like progressive muscle relaxation",
            "Follow-up assessment recommended within 1-2 weeks"
        ],
        "High Risk": [
            "Immediate consultation with a healthcare provider is recommended",
            "Consider more intensive support options",
            "Ensure regular contact with your support network",
            "Implement a daily mental wellness routine with guidance from your provider"
        ]
    }
    
    # Get emotion-specific strategies if available
    emotion_specific = []
    for insight in emotion_insights:
        if "anger" in insight.lower():
            emotion_specific.extend([
                "Practice identifying anger triggers",
                "Use timeout strategies when feeling overwhelmed",
                "Try physical outlets for frustration (e.g., exercise)"
            ])
        elif "sad" in insight.lower():
            emotion_specific.extend([
                "Schedule pleasant activities throughout your day",
                "Challenge negative thought patterns",
                "Maintain social connections even when you don't feel like it"
            ])
        elif "fear" in insight.lower() or "anxiety" in insight.lower():
            emotion_specific.extend([
                "Practice grounding techniques when feeling anxious",
                "Limit exposure to stressors when possible",
                "Challenge catastrophic thinking with evidence-based alternatives"
            ])
    
    # Combine strategies
    all_strategies = general_strategies + specific_strategies.get(risk_level, []) + emotion_specific
    
    # Deduplicate and return a reasonable number of strategies
    unique_strategies = list(dict.fromkeys(all_strategies))
    return unique_strategies[:7]  # Return maximum 7 strategies to avoid overwhelming
