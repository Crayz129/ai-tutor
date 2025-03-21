"""
Система контекстной памяти для ИИ-тьютора по математике.
"""

from typing import List, Dict, Any, Optional
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.memory.types import BaseMemory
from datetime import datetime

class MathTutorMemory(BaseMemory):
    """
    Расширенная система контекстной памяти для ИИ-тьютора по математике.
    
    Эта система памяти отслеживает:
    1. Историю диалога
    2. Текущие задачи, над которыми работает ученик
    3. Прогресс ученика по каждой задаче
    4. Ошибки, которые ученик допускал ранее
    5. Концепции, которые ученик уже изучил
    """
    
    def __init__(self, token_limit: int = 4096):
        """
        Инициализация системы памяти.
        
        Args:
            token_limit: Лимит токенов для истории диалога.
        """
        # Базовая память для истории диалога
        self.chat_memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)
        
        # Текущие задачи
        self.current_problems: Dict[str, Dict[str, Any]] = {}
        
        # Прогресс ученика
        self.student_progress: Dict[str, Any] = {
            "solved_problems": [],
            "attempted_problems": [],
            "difficult_concepts": [],
            "mastered_concepts": [],
            "common_errors": []
        }
        
        # Сессия
        self.session_info: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "session_goals": []
        }
    
    def get(self) -> Dict[str, Any]:
        """
        Получает текущее состояние памяти.
        
        Returns:
            Dict[str, Any]: Текущее состояние памяти.
        """
        # Обновление времени последней активности
        self.session_info["last_activity"] = datetime.now().isoformat()
        
        # Формирование полного состояния памяти
        memory_state = {
            "chat_history": self.chat_memory.get_all(),
            "current_problems": self.current_problems,
            "student_progress": self.student_progress,
            "session_info": self.session_info
        }
        
        return memory_state
    
    def put(self, message: Dict[str, Any]) -> None:
        """
        Добавляет сообщение в память.
        
        Args:
            message: Сообщение для добавления в память.
        """
        # Добавление сообщения в историю диалога
        self.chat_memory.put(message)
        
        # Обновление времени последней активности
        self.session_info["last_activity"] = datetime.now().isoformat()
    
    def add_problem(self, problem_id: str, problem_text: str, problem_type: str, difficulty: str) -> None:
        """
        Добавляет задачу в текущие задачи.
        
        Args:
            problem_id: Идентификатор задачи.
            problem_text: Текст задачи.
            problem_type: Тип задачи.
            difficulty: Сложность задачи.
        """
        self.current_problems[problem_id] = {
            "text": problem_text,
            "type": problem_type,
            "difficulty": difficulty,
            "status": "started",
            "attempts": 0,
            "progress": [],
            "hints_given": [],
            "errors_made": []
        }
        
        self.student_progress["attempted_problems"].append(problem_id)
    
    def update_problem_progress(self, problem_id: str, progress_update: str, is_correct: bool = None) -> None:
        """
        Обновляет прогресс по задаче.
        
        Args:
            problem_id: Идентификатор задачи.
            progress_update: Обновление прогресса.
            is_correct: Флаг правильности решения.
        """
        if problem_id not in self.current_problems:
            return
        
        # Обновление прогресса
        self.current_problems[problem_id]["progress"].append({
            "timestamp": datetime.now().isoformat(),
            "update": progress_update,
            "is_correct": is_correct
        })
        
        # Увеличение счетчика попыток
        self.current_problems[problem_id]["attempts"] += 1
        
        # Обновление статуса задачи
        if is_correct is True:
            self.current_problems[problem_id]["status"] = "solved"
            if problem_id not in self.student_progress["solved_problems"]:
                self.student_progress["solved_problems"].append(problem_id)
        elif is_correct is False:
            self.current_problems[problem_id]["status"] = "attempted"
    
    def add_hint(self, problem_id: str, hint: str) -> None:
        """
        Добавляет подсказку для задачи.
        
        Args:
            problem_id: Идентификатор задачи.
            hint: Подсказка.
        """
        if problem_id not in self.current_problems:
            return
        
        self.current_problems[problem_id]["hints_given"].append({
            "timestamp": datetime.now().isoformat(),
            "hint": hint
        })
    
    def add_error(self, problem_id: str, error: str, concept: str) -> None:
        """
        Добавляет ошибку для задачи.
        
        Args:
            problem_id: Идентификатор задачи.
            error: Описание ошибки.
            concept: Концепция, связанная с ошибкой.
        """
        if problem_id not in self.current_problems:
            return
        
        # Добавление ошибки в задачу
        self.current_problems[problem_id]["errors_made"].append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "concept": concept
        })
        
        # Добавление концепции в сложные концепции
        if concept not in self.student_progress["difficult_concepts"]:
            self.student_progress["difficult_concepts"].append(concept)
        
        # Добавление ошибки в общие ошибки
        error_entry = {
            "problem_id": problem_id,
            "error": error,
            "concept": concept,
            "timestamp": datetime.now().isoformat()
        }
        self.student_progress["common_errors"].append(error_entry)
    
    def add_mastered_concept(self, concept: str) -> None:
        """
        Добавляет освоенную концепцию.
        
        Args:
            concept: Освоенная концепция.
        """
        if concept not in self.student_progress["mastered_concepts"]:
            self.student_progress["mastered_concepts"].append(concept)
        
        # Удаление концепции из сложных концепций, если она там есть
        if concept in self.student_progress["difficult_concepts"]:
            self.student_progress["difficult_concepts"].remove(concept)
    
    def set_session_goal(self, goal: str) -> None:
        """
        Устанавливает цель сессии.
        
        Args:
            goal: Цель сессии.
        """
        self.session_info["session_goals"].append({
            "timestamp": datetime.now().isoformat(),
            "goal": goal
        })
    
    def get_context_for_problem(self, problem_id: str) -> Dict[str, Any]:
        """
        Получает контекст для задачи.
        
        Args:
            problem_id: Идентификатор задачи.
            
        Returns:
            Dict[str, Any]: Контекст для задачи.
        """
        if problem_id not in self.current_problems:
            return {}
        
        # Получение информации о задаче
        problem_info = self.current_problems[problem_id]
        
        # Получение связанных концепций
        related_concepts = []
        for error in problem_info["errors_made"]:
            if error["concept"] not in related_concepts:
                related_concepts.append(error["concept"])
        
        # Формирование контекста
        context = {
            "problem": problem_info,
            "related_concepts": related_concepts,
            "mastered_concepts": [c for c in self.student_progress["mastered_concepts"] if c in related_concepts],
            "difficult_concepts": [c for c in self.student_progress["difficult_concepts"] if c in related_concepts],
            "similar_problems": [p for p in self.student_progress["solved_problems"] if p != problem_id]
        }
        
        return context
    
    def clear(self) -> None:
        """
        Очищает память.
        """
        self.chat_memory.clear()
        self.current_problems = {}
        self.student_progress = {
            "solved_problems": [],
            "attempted_problems": [],
            "difficult_concepts": [],
            "mastered_concepts": [],
            "common_errors": []
        }
        self.session_info = {
            "start_time": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "session_goals": []
        }

    @classmethod
    def from_defaults(cls, token_limit: int = 4096):
        return cls(token_limit=token_limit)

    def get_all(self):
        return self.get()

    def reset(self):
        self.clear()

    def set(self, key, value):
        if key in {"chat_memory", "current_problems", "student_progress", "session_info"}:
            setattr(self, key, value)
        else:
            raise ValueError(f"Unknown key '{key}'")
