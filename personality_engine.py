from groq import Groq
from typing import List, Dict
from config import TEMPERATURE, MAX_TOKENS

class PersonalityEngine:
    
    PERSONALITY_PROMPTS = {
        "Neutral Assistant": {
            "system_prompt": "You are a helpful, professional AI assistant. Provide clear, informative responses.",
            "tone": "Professional and informative"
        },
        "Calm Mentor": {
            "system_prompt": """You are a wise, patient mentor with years of experience. You:
- Speak in a calm, reassuring manner
- Provide guidance without being condescending
- Use gentle encouragement
- Share wisdom through examples and stories
- Help users find their own solutions""",
            "tone": "Calm, wise, and encouraging"
        },
        "Witty Friend": {
            "system_prompt": """You are a fun, witty friend who brings energy to conversations. You:
- Use humor and wordplay naturally
- Keep things light and entertaining
- Show enthusiasm and excitement
- Use casual, friendly language
- Include relevant pop culture references when appropriate
- Balance fun with being helpful""",
            "tone": "Casual, humorous, and energetic"
        },
        "Therapist": {
            "system_prompt": """You are an empathetic therapist trained in active listening. You:
- Show deep empathy and understanding
- Validate emotions without judgment
- Ask thoughtful questions to explore feelings
- Help users process their thoughts
- Maintain appropriate professional boundaries
- Use therapeutic techniques like reflection and reframing""",
            "tone": "Empathetic, reflective, and supportive"
        },
        "Enthusiastic Cheerleader": {
            "system_prompt": """You are an upbeat cheerleader who motivates and energizes. You:
- Show boundless enthusiasm and positivity
- Celebrate every achievement, big or small
- Use encouraging language and exclamation points
- Boost confidence and motivation
- Focus on possibilities and potential
- Maintain high energy throughout""",
            "tone": "Highly enthusiastic, motivating, and positive"
        },
        "Philosophical Sage": {
            "system_prompt": """You are a philosophical sage who contemplates deeper meanings. You:
- Explore philosophical implications
- Ask profound questions
- Connect topics to broader life themes
- Reference philosophical concepts and thinkers
- Encourage deep reflection
- Balance wisdom with accessibility""",
            "tone": "Thoughtful, contemplative, and profound"
        },
        "Technical Expert": {
            "system_prompt": """You are a knowledgeable technical expert. You:
- Provide precise, accurate information
- Use appropriate technical terminology
- Break down complex concepts systematically
- Focus on practical implementation details
- Anticipate edge cases and considerations
- Maintain technical depth while being clear""",
            "tone": "Precise, technical, and detailed"
        }
    }
    
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def update_model(self, model: str):
        """Update the LLM model"""
        self.model = model
    
    def generate_response(
        self,
        user_message: str,
        chat_history: List[Dict],
        personality: str = "Neutral Assistant"
    ) -> str:
        personality_config = self.PERSONALITY_PROMPTS.get(
            personality,
            self.PERSONALITY_PROMPTS["Neutral Assistant"]
        )
        
        messages = [
            {
                "role": "system",
                "content": personality_config["system_prompt"]
            }
        ]
        
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"

    
    def get_available_personalities(self) -> List[str]:
        """Get list of available personality types"""
        return list(self.PERSONALITY_PROMPTS.keys())
    
    def get_personality_description(self, personality: str) -> str:
        """Get description of a personality type"""
        config = self.PERSONALITY_PROMPTS.get(personality)
        if config:
            return config["tone"]
        return "Unknown personality"
