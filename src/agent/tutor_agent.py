"""
ИИ-агент-тьютор для подготовки к ЕГЭ по математике.
Использует LlamaIndex для создания интерактивного тьютора, который помогает
пользователям решать задачи, не давая прямых ответов.
"""

from typing import List, Dict, Any, Optional
from llama_index.core.llms import LLM
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

class MathTutorAgent:
    """
    Агент-тьютор для подготовки к ЕГЭ по математике.
    
    Этот агент использует LlamaIndex для создания интерактивного тьютора,
    который помогает пользователям решать задачи ЕГЭ по математике,
    не давая прямых ответов, а направляя их к правильному решению.
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """
        Инициализация агента-тьютора.
        
        Args:
            llm: Языковая модель для использования. По умолчанию используется OpenAI.
        """
        # Инициализация языковой модели
        self.llm = llm or OpenAI(model="gpt-4")
        Settings.llm = self.llm
        
        # Инициализация памяти контекста
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        
        # Определение системного промпта для тьютора
        self.system_prompt = """
        Ты - тьютор по математике, помогающий ученикам подготовиться к ЕГЭ по математике.
        
        Твоя задача - помогать ученикам решать задачи, но не давать им прямых ответов.
        Вместо этого ты должен направлять их к правильному решению, задавая наводящие вопросы,
        давая подсказки и помогая исправлять ошибки.
        
        Структура ЕГЭ по математике профильного уровня:
        - Экзамен состоит из двух частей: с кратким ответом (12 заданий) и с развернутым ответом (7 заданий)
        - Всего 19 заданий, разделенных на 3 блока: алгебра, геометрия, вероятность и статистика
        - Максимальный первичный балл - 32
        - Задания с кратким ответом оцениваются в 1 балл каждое
        - Задания с развернутым ответом оцениваются от 2 до 4 баллов
        
        Для заданий с развернутым ответом требуется подробная запись решения с объяснением выполненных действий.
        
        Когда ученик задает вопрос или предоставляет решение:
        1. Определи тип задания и его сложность
        2. Если ученик просит решить задачу, не давай прямого ответа
        3. Если ученик предоставил свое решение, проверь его на ошибки
        4. Задавай наводящие вопросы, чтобы помочь ученику самостоятельно найти решение
        5. Предлагай подсказки, если ученик затрудняется
        6. Объясняй математические концепции, если это необходимо
        7. Поощряй правильные шаги и корректируй неправильные
        8. Учитывай специфику оформления решений для ЕГЭ
        
        Помни, что твоя цель - научить ученика решать задачи самостоятельно, а не решать их за него.
        """
        
        # Инициализация инструментов агента
        self.tools = self._create_tools()
        
        # Создание агента
        self.agent = ReActAgent.from_tools(
            self.tools,
            llm=self.llm,
            memory=self.memory,
            system_prompt=self.system_prompt,
            verbose=True
        )
    
    def _create_tools(self) -> List[FunctionTool]:
        """
        Создание инструментов для агента.
        
        Returns:
            List[FunctionTool]: Список инструментов для агента.
        """
        # Инструмент для анализа задачи
        analyze_problem_tool = FunctionTool.from_defaults(
            name="analyze_math_problem",
            description="Анализирует математическую задачу и определяет ее тип, сложность и ключевые концепции",
            fn=self._analyze_problem
        )
        
        # Инструмент для проверки решения
        check_solution_tool = FunctionTool.from_defaults(
            name="check_solution",
            description="Проверяет решение математической задачи на правильность",
            fn=self._check_solution
        )
        
        # Инструмент для генерации подсказок
        generate_hint_tool = FunctionTool.from_defaults(
            name="generate_hint",
            description="Генерирует подсказку для решения математической задачи",
            fn=self._generate_hint
        )
        
        return [analyze_problem_tool, check_solution_tool, generate_hint_tool]
    
    def _analyze_problem(self, problem_text: str) -> Dict[str, Any]:
        """
        Анализирует математическую задачу.
        
        Args:
            problem_text: Текст задачи.
            
        Returns:
            Dict[str, Any]: Информация о задаче.
        """
        # Заглушка для будущей реализации
        # В реальной реализации здесь будет использоваться векторное хранилище
        # для поиска похожих задач и их анализа
        return {
            "type": "Требуется анализ",
            "difficulty": "Требуется анализ",
            "key_concepts": ["Требуется анализ"],
            "similar_problems": []
        }
    
    def _check_solution(self, problem_text: str, solution_text: str) -> Dict[str, Any]:
        """
        Проверяет решение математической задачи.
        
        Args:
            problem_text: Текст задачи.
            solution_text: Текст решения.
            
        Returns:
            Dict[str, Any]: Результат проверки.
        """
        # Заглушка для будущей реализации
        # В реальной реализации здесь будет использоваться векторное хранилище
        # для поиска правильных решений и их сравнения с предоставленным решением
        return {
            "is_correct": None,  # None означает, что требуется анализ LLM
            "errors": [],
            "suggestions": []
        }
    
    def _generate_hint(self, problem_text: str, current_progress: str) -> str:
        """
        Генерирует подсказку для решения математической задачи.
        
        Args:
            problem_text: Текст задачи.
            current_progress: Текущий прогресс решения.
            
        Returns:
            str: Подсказка.
        """
        # Заглушка для будущей реализации
        # В реальной реализации здесь будет использоваться векторное хранилище
        # для поиска подходящих подсказок
        return "Подсказка будет сгенерирована на основе анализа задачи и текущего прогресса."
    
    def chat(self, message: str) -> str:
        """
        Обрабатывает сообщение пользователя и возвращает ответ.
        
        Args:
            message: Сообщение пользователя.
            
        Returns:
            str: Ответ агента.
        """
        response = self.agent.chat(message)
        return response.response
