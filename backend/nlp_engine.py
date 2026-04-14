"""
NLP Engine - Natural Language Processing Module
Handles text processing and intelligent response generation for IWO
"""

import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

try:
    from .transformer_llm import TransformerLLM
except ImportError:
    from transformer_llm import TransformerLLM

logger = logging.getLogger(__name__)


class NLPEngine:
    """Advanced Natural Language Processing engine for IWO chatbot"""

    def __init__(self, config):
        """
        Initialize advanced NLP engine for IWO

        Args:
            config: Configuration object
        """
        self.config = config
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature

        # Advanced knowledge base for IWO
        self.knowledge_base = self._load_advanced_knowledge_base()
        self.conversation_memory = {}
        self.user_preferences = {}
        self.memory_file = Path(getattr(config, "memory_file", Path(__file__).resolve().parent / "conversation_memory.json"))

        # Advanced pattern recognition
        self.intent_patterns = self._load_intent_patterns()
        self.response_templates = self._load_response_templates()

        self.transformer_llm = TransformerLLM()
        self._load_memory_file()
        logger.info("Initializing Advanced IWO NLP Engine with real AI capabilities")

    def _load_advanced_knowledge_base(self) -> Dict:
        """Load comprehensive knowledge base for advanced IWO responses"""
        return {
            "greeting": {
                "patterns": ["hello", "hi", "hey", "greetings", "good morning", "good evening", "howdy", "sup"],
                "responses": [
                    "Hello! I'm IWO, your advanced AI assistant. How can I help you today?",
                    "Hi there! I'm IWO, ready to assist you with anything you need.",
                    "Greetings! I'm IWO. What would you like to chat about?",
                    "Hey! I'm IWO, your intelligent companion. Let's have a productive conversation!",
                    "Welcome! I'm IWO, designed to help you with any questions or tasks."
                ]
            },
            "question": {
                "patterns": ["what", "how", "why", "when", "where", "who", "which", "can you"],
                "responses": [
                    "That's an excellent question! Let me provide you with a comprehensive answer:",
                    "I understand your question perfectly. Here's what I can tell you:",
                    "Great question! I'll do my best to give you a detailed and helpful response:",
                    "That's a thoughtful question. Let me share my knowledge on this topic:"
                ]
            },
            "programming": {
                "patterns": ["python", "javascript", "java", "code", "programming", "function", "class", "variable", "debug", "error", "algorithm"],
                "responses": [
                    "As an advanced AI, I can help you with programming! What language are you working with?",
                    "Programming is one of my core competencies. Let me assist you with your code.",
                    "I can help with debugging, algorithms, best practices, and code optimization. What's your question?",
                    "I'm well-versed in multiple programming languages. How can I help you with your project?"
                ],
                "knowledge": {
                    "python": "Python is a versatile, high-level language known for its simplicity and readability. It's excellent for web development, data science, AI, automation, and rapid prototyping.",
                    "javascript": "JavaScript is the language of the web! It runs in browsers and servers (Node.js), making it perfect for full-stack development, interactive websites, and modern applications.",
                    "debugging": "When debugging, systematically check: 1) Syntax errors, 2) Variable scoping, 3) Logic flow, 4) Input validation, 5) Error handling. Use logging and breakpoints effectively."
                }
            },
            "math": {
                "patterns": ["math", "calculate", "equation", "algebra", "geometry", "calculus", "number", "solve", "formula", "theorem"],
                "responses": [
                    "Mathematics is fascinating! I can help with calculations, explanations, and problem-solving.",
                    "I love mathematical challenges. Let me help you work through this step by step.",
                    "Mathematical concepts can be complex, but I can break them down clearly for you.",
                    "I'm equipped to handle various mathematical problems and explanations."
                ],
                "knowledge": {
                    "algebra": "Algebra deals with symbols and rules for manipulating mathematical expressions. It's fundamental to many fields including physics, engineering, and computer science.",
                    "geometry": "Geometry studies shapes, sizes, and properties of space. It's essential in architecture, computer graphics, physics, and many engineering disciplines.",
                    "calculus": "Calculus studies continuous change through derivatives and integrals. It's crucial for understanding rates of change, optimization, and many scientific phenomena."
                }
            },
            "science": {
                "patterns": ["physics", "chemistry", "biology", "science", "atom", "molecule", "theory", "experiment", "research", "quantum"],
                "responses": [
                    "Science is amazing! I can help explain concepts in physics, chemistry, biology, and more.",
                    "Scientific questions are always interesting. Let me provide some insights.",
                    "I can help you understand scientific principles and current discoveries.",
                    "I'm knowledgeable about various scientific fields. What interests you?"
                ],
                "knowledge": {
                    "physics": "Physics explores the fundamental laws of nature, from quantum mechanics to cosmology. It explains how everything in the universe works.",
                    "chemistry": "Chemistry studies matter, its properties, and transformations. It's the science of atoms, molecules, and chemical reactions.",
                    "biology": "Biology studies living organisms and their interactions with the environment. It encompasses everything from cells to ecosystems."
                }
            },
            "technology": {
                "patterns": ["computer", "software", "hardware", "internet", "ai", "machine learning", "robot", "automation", "tech", "digital"],
                "responses": [
                    "Technology is constantly evolving! What aspect interests you most?",
                    "I can help you understand the latest in tech, from AI to hardware.",
                    "Technology questions are right up my alley. What would you like to know?",
                    "I'm well-informed about current technological trends and developments."
                ],
                "knowledge": {
                    "ai": "Artificial Intelligence uses algorithms and data to perform tasks that typically require human intelligence, like pattern recognition, decision-making, and language understanding.",
                    "machine_learning": "Machine Learning is a subset of AI where systems learn from data to improve performance without being explicitly programmed for every scenario.",
                    "automation": "Automation uses technology to perform tasks with minimal human intervention, increasing efficiency and reducing errors in various industries."
                }
            },
            "help": {
                "patterns": ["help", "what can you do", "capabilities", "features", "assist", "commands", "guide"],
                "responses": [
                    "I'm IWO, an advanced AI assistant with extensive capabilities! I can help with:\n• Programming and coding assistance\n• Mathematics and problem-solving\n• Science and technology explanations\n• Creative writing and brainstorming\n• General knowledge and research\n• Learning and education\n• And much more!",
                    "My capabilities are extensive! I can assist with programming, mathematics, science, technology, creative tasks, and general knowledge. What interests you?",
                    "I'm designed to be your comprehensive AI companion. I can help with coding, math, science, writing, research, and many other intellectual tasks."
                ]
            },
            "acknowledgment": {
                "patterns": ["ok", "okay", "alright", "sure", "got it", "sounds good", "thanks", "thank you"],
                "responses": [
                    "Understood. Whenever you're ready, tell me what you'd like help with next.",
                    "Okay, I'm here when you want to continue or ask a new question.",
                    "Got it. If you want, I can help you with something specific now."
                ]
            },
            "creative": {
                "patterns": ["write", "story", "poem", "creative", "design", "art", "music", "brainstorm", "imagine", "create"],
                "responses": [
                    "Creativity is one of my strengths! I can help with writing, brainstorming, and innovative ideas.",
                    "I love creative tasks. Let me help you with your project.",
                    "Whether it's writing, design, or problem-solving, I can bring some creativity to the table.",
                    "I'm equipped for various creative endeavors. What would you like to create?"
                ]
            },
            "learning": {
                "patterns": ["learn", "teach", "study", "understand", "explain", "tutorial", "lesson", "education"],
                "responses": [
                    "Learning is fundamental! I can help you understand complex topics and concepts.",
                    "I'm an excellent teacher. Let me break down this topic for you.",
                    "Education is important. I can provide clear explanations and examples.",
                    "I can help you learn new concepts with detailed explanations and examples."
                ]
            },
            "philosophy": {
                "patterns": ["philosophy", "meaning", "life", "existence", "consciousness", "ethics", "morality", "reality"],
                "responses": [
                    "Philosophy explores fundamental questions about existence, knowledge, and ethics.",
                    "Deep philosophical questions are fascinating. Let me share some insights.",
                    "I'm equipped to discuss philosophical concepts and ideas.",
                    "Philosophy helps us understand the nature of reality and our place in it."
                ]
            },
            "identity": {
                "patterns": ["who are you", "what are you", "are you ai", "are you artificial intelligence", "what is your name", "who is iwo", "tell me about yourself", "about yourself", "real ai", "really real ai", "truly real ai"],
                "responses": [
                    "I'm IWO, a fully self-contained local AI assistant built to think, learn, and assist without any external API or cloud dependency. I run entirely inside this application using local NLP, memory, and knowledge base systems.",
                    "Hello! I'm IWO, your advanced local AI companion. I operate completely offline as a self-contained intelligence engine designed to help with programming, math, science, creative writing, learning, and more.",
                    "I'm IWO, a sophisticated local AI assistant. My intelligence is generated in this codebase, not from an external service. I can help with coding, problem-solving, explanations, and creative tasks while preserving full offline independence.",
                    "Hi! I'm IWO, a real local AI designed to be your comprehensive assistant. I can engage in meaningful conversations, provide detailed explanations, help with programming and math, and assist with many intellectual tasks without calling any API."
                ]
            },
            "default": {
                "responses": [
                    "That's a fascinating topic! I'd love to explore it with you.",
                    "I appreciate you sharing that. Let me offer some thoughts.",
                    "That's an interesting point. Here's my take on it:",
                    "I can help you explore this further. What specific aspect interests you?",
                    "That's worth discussing! Let me provide some insights."
                ]
            }
        }

    def _load_intent_patterns(self) -> Dict:
        """Load advanced intent recognition patterns"""
        return {
            "question": re.compile(r'\b(what|how|why|when|where|who|which|can you|do you|are you)\b', re.IGNORECASE),
            "programming": re.compile(r'\b(python|javascript|java|c\+\+|code|coding|programming|function|class|variable|debug|error|algorithm)\b', re.IGNORECASE),
            "math": re.compile(r'\b(math|calculate|equation|algebra|geometry|calculus|number|solve|formula|theorem)\b', re.IGNORECASE),
            "science": re.compile(r'\b(physics|chemistry|biology|science|atom|molecule|theory|experiment|research|quantum)\b', re.IGNORECASE),
            "technology": re.compile(r'\b(computer|software|hardware|internet|ai|machine learning|machine-learning|robot|automation|tech|digital)\b', re.IGNORECASE),
            "creative": re.compile(r'\b(write|story|poem|creative|design|art|music|brainstorm|imagine|create)\b', re.IGNORECASE),
            "learning": re.compile(r'\b(learn|teach|study|understand|explain|tutorial|lesson|education)\b', re.IGNORECASE),
            "philosophy": re.compile(r'\b(philosophy|meaning|life|existence|consciousness|ethics|morality|reality)\b', re.IGNORECASE),
            "greeting": re.compile(r'\b(hello|hi|hey|greetings|good morning|good evening|howdy|sup|welcome)\b', re.IGNORECASE),
            "identity": re.compile(r'\b(who are you|what are you|are you ai|are you artificial intelligence|what is your name|who is iwo|tell me about yourself|about yourself|real ai|really real ai|truly real ai)\b', re.IGNORECASE),
            "help": re.compile(r'\b(help|what can you do|capabilities|features|assist|commands|guide)\b', re.IGNORECASE),
            "acknowledgment": re.compile(r'\b(ok|okay|alright|sure|got it|sounds good|thanks|thank you)\b', re.IGNORECASE)
        }

    def _load_response_templates(self) -> Dict:
        """Load sophisticated response templates"""
        return {
            "follow_up": [
                "Is there anything else you'd like me to cover? 😊",
                "Would you like more detail on that? 🤔",
                "Anything else you want to explore? ✨",
                "Is there another angle you want me to take? 👍"
            ],
            "clarification": [
                "Can you tell me a little more so I can answer precisely? 😊",
                "I want to make sure I understood you correctly. Could you elaborate? 🤔",
                "A bit more context would help me give you the best reply. 🧠",
                "Could you clarify what you mean by that? 🙋"
            ],
            "clarify_question": [
                "That's a good question. Can you tell me a bit more about what you want to know? 🤔",
                "What exactly are you asking about? A little more detail will help me answer it. 📝",
                "I'm happy to explain - can you narrow that down a bit? 😊",
                "Can you give me a little more context so I can answer clearly? 👍"
            ],
            "acknowledgment": [
                "Got it. 👍",
                "That makes sense. 😊",
                "I hear you. 👂",
                "Understood. ✅"
            ],
            "tone": {
                "friendly": [
                    "Sure, let's walk through that. 😊",
                    "Absolutely, here's what I think. 💡",
                    "Okay, I can help with that. 👍",
                    "Let's take a closer look. 🔍"
                ],
                "concise": [
                    "Here's a quick answer: ✅",
                    "In short: ⏱️",
                    "To keep it brief: ✨",
                    "Here's the main point: 🧾"
                ],
                "explanatory": [
                    "Here's how I see it: 🧠",
                    "I can explain that in more detail: 📘",
                    "Let me break that down for you: 🔧",
                    "Here's the explanation: 📝"
                ]
            }
        }

    def generate_response(self, user_input: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Generate intelligent response using advanced AI capabilities

        Args:
            user_input: User message
            conversation_history: Previous conversation messages

        Returns:
            Generated response
        """
        try:
            # Advanced intent analysis
            intent, confidence = self._analyze_intent_advanced(user_input)

            # Get conversation context and style
            context = self._extract_context(conversation_history)
            style = self.adapt_response_style(user_input, context)

            response = self.process_input(user_input, intent, confidence, context, style)

            # Add follow-up if appropriate
            if random.random() < 0.2 and len(user_input.split()) > 4 and intent not in {"identity", "greeting", "help", "acknowledgment"}:
                follow_up = random.choice(self.response_templates["follow_up"])
                response += f"\n\n{follow_up}"

            # Update conversation memory
            self._update_memory(user_input, response, intent)

            logger.info(f"IWO generated advanced response for intent: {intent} (confidence: {confidence:.2f}, style={style})")
            return response

        except Exception as e:
            logger.error(f"Error in advanced response generation: {str(e)}")
            return "I'm here to help. What would you like to know or discuss?"

    def process_input(self, user_input: str, intent: str, confidence: float, context: Dict, style: str) -> str:
        """Decide whether to use the knowledge-base templates or the transformer LLM."""
        normalized_text = re.sub(r'[^a-z0-9\s]', '', user_input.lower())
        if intent == "identity":
            return self._get_identity_response()

        if "makes sense" in normalized_text:
            return "That means I understand your point and I think the idea is clear. If you'd like, tell me which part you'd like me to explain further."

        if self._is_ambiguous_question(user_input):
            return random.choice(self.response_templates["clarify_question"])

        if intent in {"greeting", "help"}:
            return self._generate_template_response(intent)

        if intent == "acknowledgment":
            return random.choice(self.response_templates["acknowledgment"])

        if intent == "question" and confidence < 0.5:
            return random.choice(self.response_templates["clarification"])

        if self._should_use_transformer(user_input, intent, confidence, context):
            return self._generate_llm_response(user_input, context, style)

        return self._generate_template_response(intent)

    def _generate_template_response(self, intent: str) -> str:
        intent_data = self.knowledge_base.get(intent, self.knowledge_base["default"])
        if isinstance(intent_data, dict) and intent_data.get("responses"):
            return random.choice(intent_data["responses"])
        return random.choice(self.knowledge_base["default"]["responses"])

    def _get_identity_response(self) -> str:
        identity_responses = self.knowledge_base.get("identity", {}).get("responses", [])
        if identity_responses:
            return identity_responses[0]
        return "I'm IWO, a local AI chatbot built to run without external APIs."

    def _should_use_transformer(self, user_input: str, intent: str, confidence: float, context: Dict) -> bool:
        normalized_text = re.sub(r'[^a-z0-9\s]', '', user_input.lower())
        tokens = normalized_text.split()

        if intent in {"greeting", "help", "acknowledgment", "identity"}:
            return False

        important_short_question = any(phrase in normalized_text for phrase in ["makes sense", "mean", "meaning", "why not", "why does", "how come"])
        if len(tokens) <= 3 and intent in {"question", "default"} and not important_short_question:
            return False

        if intent in {"programming", "math", "science", "technology", "creative", "learning", "philosophy"} and confidence < 0.6:
            return False

        if len(tokens) <= 5 and any(q in normalized_text for q in ["what", "why", "how", "when", "where", "which"]) and not important_short_question:
            return False

        if "information" in normalized_text and len(tokens) < 8:
            return False

        if user_input.lower().count("information") >= 3 or user_input.lower().count("detail") >= 3:
            return False

        if len(tokens) >= 8:
            unique_ratio = len(set(tokens)) / max(1, len(tokens))
            if unique_ratio < 0.45:
                return False

        if "[bos]" in user_input.lower() or "[eos]" in user_input.lower() or "<bos>" in user_input.lower() or "<eos>" in user_input.lower():
            return False

        return True

    def _build_llm_prompt(self, user_input: str, context: Dict, style: str) -> str:
        prompt = f"User: {user_input}\nAI:"
        return prompt

    def _generate_llm_response(self, user_input: str, context: Dict, style: str) -> str:
        prompt = self._build_llm_prompt(user_input, context, style)
        llm_output = self.transformer_llm.generate(prompt, max_tokens=64, temperature=self.temperature)
        if not llm_output.strip():
            return "I can help with that. Please tell me more about what you need."
        return llm_output

    def _analyze_intent_advanced(self, text: str) -> Tuple[str, float]:
        """Advanced intent analysis with confidence scoring"""
        text_lower = text.lower()
        scores = {}

        # Priority check for explicit identity and local AI questions
        identity_phrases = [
            "who are you",
            "what are you",
            "are you ai",
            "are you artificial intelligence",
            "what is your name",
            "who is iwo",
            "tell me about yourself",
            "about yourself",
            "real ai",
            "really real ai",
            "truly real ai",
            "do you use an api",
            "do you use an external api",
            "are you using an api",
            "do you use api",
            "api"
        ]

        normalized_text = re.sub(r'[^a-z0-9\s]', '', text_lower)
        for phrase in identity_phrases:
            if phrase in normalized_text:
                return "identity", 0.95

        tokens = normalized_text.split()
        if "ai" in tokens and any(word in tokens for word in ["real", "really", "realy", "true", "truly", "authentic"]) and any(word in tokens for word in ["you", "are", "is", "am", "this", "that"]):
            return "identity", 0.95

        if "ai" in tokens and "you" in tokens and len(tokens) <= 6:
            return "identity", 0.9

        # Special handling for simple grammar and direct requests
        if "can you" in normalized_text and any(word in normalized_text for word in ["code", "coding", "program", "debug", "python", "javascript", "java"]):
            return "programming", 0.9
        if "can you" in normalized_text and any(word in normalized_text for word in ["help", "assist", "support"]):
            return "help", 0.85

        # Check against all intent patterns
        for intent, pattern in self.intent_patterns.items():
            matches = len(pattern.findall(text_lower))
            if matches > 0:
                # Calculate confidence based on matches and text length
                confidence = min(matches * 0.3 + (len(text) / 100), 1.0)
                scores[intent] = confidence

        # Special handling for questions
        if '?' in text or text.startswith(('what', 'how', 'why', 'when', 'where', 'who', 'which')):
            scores['question'] = scores.get('question', 0) + 0.4

        # Return highest confidence intent
        if scores:
            best_intent = max(scores, key=scores.get)
            return best_intent, scores[best_intent]

        return "default", 0.5

    def _extract_context(self, conversation_history: Optional[List[Dict]] = None) -> Dict:
        """Extract conversation context for better responses"""
        context = {
            "recent_topics": [],
            "user_mood": "neutral",
            "conversation_length": 0,
            "last_intent": None
        }

        if not conversation_history:
            return context

        context["conversation_length"] = len(conversation_history)

        # Analyze recent messages for topics and patterns
        recent_messages = conversation_history[-5:]  # Last 5 messages
        topics = set()

        for msg in recent_messages:
            if msg.get('user_message'):
                # Extract keywords without punctuation
                tokens = re.findall(r"[a-zA-Z]{5,}", msg['user_message'].lower())
                topics.update(tokens)

            if msg.get('intent'):
                context["last_intent"] = msg['intent']

        context["recent_topics"] = list(topics)[:3]  # Top 3 topics

        return context

    def _generate_advanced_response(self, user_input: str, intent: str, confidence: float, context: Dict, style: str) -> str:
        """Generate sophisticated response based on intent, confidence, context, and style"""
        intent_data = self.knowledge_base.get(intent, self.knowledge_base["default"])

        if isinstance(intent_data, dict) and "responses" in intent_data:
            responses = intent_data["responses"]
        else:
            responses = self.knowledge_base["default"]["responses"]

        if context["conversation_length"] > 10:
            response_idx = self._stable_hash_to_index(user_input + str(context["conversation_length"]), len(responses))
        else:
            response_idx = self._stable_hash_to_index(user_input, len(responses))

        if intent == "question" and self._is_ambiguous_question(user_input):
            return random.choice(self.response_templates["clarify_question"])

        base_response = responses[response_idx]

        if isinstance(intent_data, dict) and "knowledge" in intent_data:
            knowledge = self._get_relevant_knowledge(user_input, intent_data["knowledge"])
            if knowledge:
                base_response += f"\n\n{knowledge}"

        if intent != "identity" and context["conversation_length"] > 1 and context["recent_topics"]:
            context_phrase = f" I also noticed we recently discussed {', '.join(context['recent_topics'][:2])}."
            if len(base_response + context_phrase) < 500:
                base_response += context_phrase

        if confidence > 0.7:
            base_response = self._add_personality(base_response, "confident")
        elif confidence < 0.3:
            base_response = self._add_personality(base_response, "exploratory")

        base_response = self._apply_response_style(base_response, style)
        return base_response

    def _get_relevant_knowledge(self, user_input: str, knowledge_dict: Dict) -> Optional[str]:
        """Get relevant knowledge snippet based on user input"""
        user_lower = user_input.lower()

        for key, knowledge in knowledge_dict.items():
            if key in user_lower:
                return knowledge

        return None

    def _add_personality(self, response: str, personality_type: str) -> str:
        """Add personality traits to responses"""
        if personality_type == "confident":
            return response
        elif personality_type == "exploratory":
            return response + " I'm curious to learn more about what you're thinking."
        elif personality_type == "helpful":
            return "I'd be happy to help! " + response

        return response

    def _is_ambiguous_question(self, user_input: str) -> bool:
        """Detect short or vague question prompts that need clarification"""
        text = user_input.strip().lower()
        short_questions = {"what", "why", "how", "who", "when", "where", "which", "why?", "how?", "who?", "when?", "where?", "what?", "which?"}
        if text in short_questions:
            return True

        tokens = re.findall(r"[a-zA-Z]+", text)
        return len(tokens) == 1 and text.endswith("?")

    def _apply_response_style(self, response: str, style: str) -> str:
        """Apply a natural conversational tone to the response"""
        prefixes = self.response_templates.get("tone", {}).get(style, [])
        if prefixes and response:
            prefix = random.choice(prefixes)
            return f"{prefix} {response}"
        return response

    def adapt_response_style(self, user_input: str, context: Dict) -> str:
        """Choose a natural response style based on the input and conversation context"""
        lower_text = user_input.lower()
        if len(lower_text.split()) < 5 and lower_text.endswith("?"):
            return "concise"
        if any(word in lower_text for word in ["explain", "describe", "why", "how", "what"]):
            return "explanatory"
        if context["conversation_length"] > 8:
            return "friendly"
        return "friendly"

    def _update_memory(self, user_input: str, response: str, intent: str):
        """Update conversation memory for learning"""
        memory_key = f"session_{datetime.now().strftime('%Y%m%d')}"

        if memory_key not in self.conversation_memory:
            self.conversation_memory[memory_key] = []

        self.conversation_memory[memory_key].append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "intent": intent
        })

        if len(self.conversation_memory[memory_key]) > 50:
            self.conversation_memory[memory_key] = self.conversation_memory[memory_key][-50:]

        self._save_memory_file()

    def _load_memory_file(self):
        """Load persisted memory from disk if available"""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            if self.memory_file.exists():
                with self.memory_file.open("r", encoding="utf-8") as f:
                    self.conversation_memory = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load conversation memory file: {e}")

    def _save_memory_file(self):
        """Save conversation memory to disk"""
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with self.memory_file.open("w", encoding="utf-8") as f:
                json.dump(self.conversation_memory, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save conversation memory file: {e}")

    def _stable_hash_value(self, text: str) -> int:
        """Deterministic hash for stable response selection"""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(digest, 16)

    def _stable_hash_to_index(self, text: str, modulus: int) -> int:
        """Map deterministic hash to a response index"""
        if modulus <= 0:
            return 0
        return self._stable_hash_value(text) % modulus

    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple word-based tokenization
        return text.lower().split()

    def get_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for text

        Args:
            text: Input text

        Returns:
            Embedding vector as list
        """
        tokens = self.tokenize(text)

        # Simple hash-based embedding
        embedding = [self._stable_hash_to_index(token, 100) / 100.0 for token in tokens]

        # Pad or truncate to fixed size
        if len(embedding) < self.max_tokens:
            embedding.extend([0.0] * (self.max_tokens - len(embedding)))
        else:
            embedding = embedding[:self.max_tokens]

        return embedding

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        mag1 = sum(a * a for a in emb1) ** 0.5
        mag2 = sum(b * b for b in emb2) ** 0.5

        similarity = dot_product / (mag1 * mag2 + 1e-10)
        return float(similarity)

    def detect_intent(self, text: str) -> str:
        """
        Detect user intent from text (legacy method)

        Args:
            text: User input

        Returns:
            Intent type
        """
        return self._analyze_intent_advanced(text)[0]

    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Simple text summarization"""
        text = text.strip()
        if len(text) <= max_length:
            return text

        sentences = re.split(r'(?<=[.!?])\s+', text)
        summary_parts = []
        total_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if total_length + len(sentence) + len(summary_parts) <= max_length:
                summary_parts.append(sentence)
                total_length += len(sentence)
            else:
                break

        if summary_parts:
            summary = " ".join(summary_parts)
            if len(summary) > max_length:
                summary = summary[:max_length].rstrip()
            return summary

        return text[:max_length].rstrip() + "..."

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Advanced entity extraction from text"""
        entities = {
            "keywords": [],
            "questions": [],
            "topics": [],
            "numbers": [],
            "code_snippets": [],
            "urls": []
        }

        # Extract keywords (longer words)
        tokens = self.tokenize(text)
        entities["keywords"] = [t for t in tokens if len(t) > 5 and t.isalpha()]

        # Extract questions
        if '?' in text:
            entities["questions"].append(text)

        # Extract numbers
        number_pattern = re.compile(r'\b\d+\.?\d*\b')
        entities["numbers"] = number_pattern.findall(text)

        # Extract code-like patterns
        code_pattern = re.compile(r'`[^`]+`')
        entities["code_snippets"] = code_pattern.findall(text)

        # Extract URLs
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        entities["urls"] = url_pattern.findall(text)

        # Determine topics based on content
        topics = []
        text_lower = text.lower()
        if any(word in text_lower for word in ['python', 'javascript', 'java', 'programming', 'code']):
            topics.append("programming")
        if any(word in text_lower for word in ['math', 'calculate', 'equation', 'algebra']):
            topics.append("mathematics")
        if any(word in text_lower for word in ['physics', 'chemistry', 'biology', 'science']):
            topics.append("science")
        if any(word in text_lower for word in ['ai', 'machine learning', 'robot', 'technology']):
            topics.append("technology")

        entities["topics"] = topics

        return entities

    def analyze_sentiment(self, text: str) -> str:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst', 'suck']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def get_response(self, text: str) -> str:
        """
        Main response method that decides between knowledge_base and transformer_llm.
        This is the fuel and transmission that connects input to the AI engine.
        """
        try:
            # 1. Check against specific patterns (fast and economical)
            # Only use templates for very specific, short greetings/help requests
            text_lower = text.lower().strip()

            # Check for exact greeting matches
            if any(re.search(rf"\b{re.escape(p)}\b", text_lower) for p in ["hello", "hi", "hey", "greetings", "good morning", "good evening", "howdy", "sup"]):
                return random.choice(self.knowledge_base["greeting"]["responses"])

            # Check for help/capability questions
            if any(phrase in text_lower for phrase in ["what can you do", "help", "capabilities", "features", "assist", "commands", "guide"]):
                return random.choice(self.knowledge_base["help"]["responses"])

            # Check for identity questions
            if any(phrase in text_lower for phrase in ["who are you", "what are you", "are you ai", "what is your name", "who is iwo", "tell me about yourself"]):
                return random.choice(self.knowledge_base["identity"]["responses"])

            # 2. For everything else - use the real AI with memory context
            return self._generate_with_memory(text)

        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            # 4. Error handling (Fallback) - if transformer fails, fall back to general help
            return random.choice(self.knowledge_base.get("help", {}).get("responses",
                ["I'm here to help. What would you like to know or discuss?"]))

    def _generate_with_memory(self, text: str) -> str:
        """Generate response using transformer with conversation memory context."""
        try:
            # 3. Memory management in Prompt - inject conversation_memory
            memory_context = self._build_memory_context()
            enhanced_prompt = f"{memory_context}\nUser: {text}\nAI:"

            # Use transformer with memory-enhanced prompt
            response = self.transformer_llm.generate(enhanced_prompt, max_tokens=self.max_tokens, temperature=self.temperature)

            if not response or len(response.strip()) < 5:
                # Fallback if transformer gives empty/short response
                return random.choice(self.knowledge_base.get("default", {}).get("responses",
                    ["That's interesting! Tell me more about it."]))

            return response

        except Exception as e:
            logger.error(f"Error in transformer generation: {str(e)}")
            # Fallback to template responses
            return random.choice(self.knowledge_base.get("help", {}).get("responses",
                ["I can help with that. Please tell me more."]))

    def _build_memory_context(self) -> str:
        """Build context string from recent conversation memory."""
        if not self.conversation_memory:
            return ""

        # Get most recent session
        recent_session_key = max(self.conversation_memory.keys(),
                                key=lambda k: self.conversation_memory[k][-1]["timestamp"] if self.conversation_memory[k] else "")

        recent_messages = self.conversation_memory.get(recent_session_key, [])[-3:]  # Last 3 exchanges

        context_parts = []
        for msg in recent_messages:
            context_parts.append(f"Previous: {msg.get('user_input', '')}")
            context_parts.append(f"AI: {msg.get('response', '')}")

        return "\n".join(context_parts) if context_parts else ""

    def load_pretrained_weights(self, weights_path: str):
        """
        Load pre-trained weights into the transformer model.
        This gives the AI real intelligence instead of random weights.
        """
        try:
            self.transformer_llm.load_weights(weights_path)
            logger.info(f"Successfully loaded pre-trained weights from {weights_path}")
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights: {str(e)}. Using random initialization.")

    def generate_follow_up_questions(self, user_input: str, intent: str) -> List[str]:
        """Generate contextual follow-up questions"""
        follow_ups = {
            "programming": [
                "What programming language are you most comfortable with?",
                "Are you working on a specific project or learning?",
                "Do you need help with debugging or new features?"
            ],
            "math": [
                "What level of mathematics are you studying?",
                "Are you working on a specific problem?",
                "Would you like me to show you the steps?"
            ],
            "science": [
                "Which scientific field interests you most?",
                "Are you studying this for school or personal interest?",
                "Would you like me to explain related concepts?"
            ],
            "technology": [
                "What aspect of technology are you most curious about?",
                "Are you interested in current trends or fundamentals?",
                "Do you have a specific technology question?"
            ]
        }

        return follow_ups.get(intent, [
            "Would you like me to elaborate on this topic?",
            "Is there a specific aspect you'd like to explore further?",
            "Do you have any related questions?"
        ])

    def learn_from_interaction(self, user_input: str, user_feedback: Optional[str] = None):
        """Learn from user interactions to improve responses"""
        # This is a basic learning system - could be expanded
        if user_feedback:
            # Store successful interaction patterns
            self.user_preferences[user_input.lower()] = {
                "feedback": user_feedback,
                "timestamp": datetime.now().isoformat()
            }

    def train_transformer(self, texts: List[str], epochs: int = 1, lr: float = 1e-3, max_length: int = 64) -> List[float]:
        """Train the local transformer model on provided text data."""
        if not hasattr(self, 'transformer_llm'):
            raise RuntimeError("Transformer LLM is not initialized.")
        return self.transformer_llm.train_on_texts(texts, epochs=epochs, lr=lr, max_length=max_length)

    def train_transformer_from_file(self, path: str, epochs: int = 1, lr: float = 1e-3, max_length: int = 64, sample_limit: Optional[int] = None) -> List[float]:
        """Train the transformer from a local text file."""
        if not hasattr(self, 'transformer_llm'):
            raise RuntimeError("Transformer LLM is not initialized.")
        return self.transformer_llm.train_from_file(path, epochs=epochs, lr=lr, max_length=max_length, sample_limit=sample_limit)

    def get_conversation_insights(self) -> Dict:
        """Get insights from conversation history"""
        insights = {
            "total_interactions": 0,
            "common_topics": [],
            "avg_response_length": 0,
            "user_satisfaction": "unknown"
        }

        total_length = 0
        topics = {}

        for session, messages in self.conversation_memory.items():
            insights["total_interactions"] += len(messages)

            for msg in messages:
                total_length += len(msg.get("response", ""))

                # Count topics
                intent = msg.get("intent", "")
                topics[intent] = topics.get(intent, 0) + 1

        if insights["total_interactions"] > 0:
            insights["avg_response_length"] = total_length / insights["total_interactions"]
            insights["common_topics"] = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]

        return insights