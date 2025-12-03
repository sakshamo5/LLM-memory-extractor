import json
from groq import Groq
from typing import List, Dict
from config import TEMPERATURE, MAX_TOKENS

class MemoryExtractor:
    
    def __init__(self, api_key: str, model: str = "openai/gpt-oss-20b"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def update_model(self, model: str):
        """Update the LLM model"""
        self.model = model
    
    def extract_memory(self, chat_history: List[Dict]) -> Dict:
        conversation_text = self._format_chat_history(chat_history)
        
        extraction_prompt = f"""You are a memory extraction AI. Analyze the following conversation and extract:

    1. **User Preferences**: Likes, dislikes, interests, hobbies, work preferences, communication style preferences
    2. **Emotional Patterns**: Recurring emotions, stress triggers, happiness sources, communication tone, energy levels
    3. **Facts to Remember**: Personal facts, important dates, relationships, goals, challenges, background information

    Conversation:
    {conversation_text}

    Provide your analysis in the following JSON format:
    {{
        "user_preferences": [
            "preference 1",
            "preference 2"
        ],
        "emotional_patterns": [
            "pattern 1",
            "pattern 2"
        ],
        "facts_to_remember": [
            "fact 1",
            "fact 2"
        ],
        "conversation_summary": "Brief summary of the conversation",
        "dominant_topics": ["topic1", "topic2"],
        "user_personality_traits": ["trait1", "trait2"]
    }}

    Extract as much detail as possible. Each item should be specific and actionable. Return ONLY valid JSON."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing conversations and extracting structured memory data. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": extraction_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=MAX_TOKENS
            )
            
            result = response.choices[0].message.content.strip()
            
            if result.startswith("```json"):
                result = result[7:]
                if result.endswith("```"):
                    result = result[:-3]
            elif result.startswith("```json"):
                result = result[3:]
                if result.endswith("```"):
                    result = result[:-3]
            
            result = result.strip()
            memory_data = json.loads(result)
            return memory_data
            
        except json.JSONDecodeError as e:
            return {
                "user_preferences": ["Unable to parse preferences"],
                "emotional_patterns": ["Unable to parse emotional patterns"],
                "facts_to_remember": ["Unable to parse facts"],
                "conversation_summary": "Extraction failed",
                "dominant_topics": [],
                "user_personality_traits": [],
                "error": str(e)
            }
        except Exception as e:
            return {
                "user_preferences": [],
                "emotional_patterns": [],
                "facts_to_remember": [],
                "conversation_summary": "Error during extraction",
                "dominant_topics": [],
                "user_personality_traits": [],
                "error": str(e)
            }

    
    def _format_chat_history(self, chat_history: List[Dict]) -> str:
        formatted = []
        for msg in chat_history:
            role = msg['role'].upper()
            content = msg['content']
            formatted.append(f"{role}: {content}")
        return "\n\n".join(formatted)
    
    def extract_preferences_only(self, chat_history: List[Dict]) -> List[str]:
        memory = self.extract_memory(chat_history)
        return memory.get("user_preferences", [])
    
    def extract_emotional_patterns_only(self, chat_history: List[Dict]) -> List[str]:
        memory = self.extract_memory(chat_history)
        return memory.get("emotional_patterns", [])
    
    def extract_facts_only(self, chat_history: List[Dict]) -> List[str]:
        memory = self.extract_memory(chat_history)
        return memory.get("facts_to_remember", [])
